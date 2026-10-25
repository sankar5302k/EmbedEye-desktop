"""Strict GPU Memory Manager for Local Photo Search on RTX 3050 4GB.

Enforces that only ONE heavy model lives on GPU at any time during indexing,
while allowing CLIP and Cross-Encoder Reranker to coexist at search time (~1.5GB total).
Florence-2 NEVER coexists with any other model on GPU.
"""

from __future__ import annotations

import gc
import logging
import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

import src.config as cfg

logger = logging.getLogger("photo_search.gpu_manager")


class ModelManager:
    """Manages lifecycle and strict GPU memory allocation of AI models."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or cfg.get_device()
        self._lock = threading.RLock()
        self._loaders: Dict[str, Callable[[], Any]] = {}
        self._unloaders: Dict[str, Callable[[Any], None]] = {}
        self._active_models: Dict[str, Any] = {}
        self._peak_vram_mb: float = 0.0

    def register_loader(
        self,
        name: str,
        load_fn: Callable[[], Any],
        unload_fn: Optional[Callable[[Any], None]] = None,
    ) -> None:
        """Register model loader and optional unloader callbacks."""
        with self._lock:
            self._loaders[name] = load_fn
            if unload_fn:
                self._unloaders[name] = unload_fn

    def get_vram_stats(self) -> Dict[str, float]:
        """Query current allocated and reserved CUDA VRAM in MB."""
        allocated = 0.0
        reserved = 0.0
        total = 0.0
        if self.device == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                    total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                    if allocated > self._peak_vram_mb:
                        self._peak_vram_mb = allocated
            except Exception as e:
                logger.debug(f"Could not read CUDA VRAM: {e}")
        return {
            "allocated_mb": round(allocated, 1),
            "reserved_mb": round(reserved, 1),
            "total_mb": round(total, 1),
            "free_mb": round(max(0.0, total - allocated), 1),
        }

    def vram_report(self) -> Dict[str, Any]:
        """Return structured VRAM and resident model status for logs and UI."""
        stats = self.get_vram_stats()
        dev_name = "CPU"
        if self.device == "cuda":
            try:
                import torch

                dev_name = torch.cuda.get_device_name(0)
            except Exception:
                dev_name = "CUDA (Device 0)"

        with self._lock:
            resident = list(self._active_models.keys())

        resident_str = ", ".join(resident) if resident else "None (idle)"
        status_text = (
            f"Device: {dev_name} | Resident Models: [{resident_str}] | "
            f"VRAM: {stats['allocated_mb']}MB allocated / {stats['reserved_mb']}MB reserved "
            f"(Total: {stats['total_mb']}MB)"
        )

        return {
            "device": self.device,
            "device_name": dev_name,
            "resident_models": resident,
            "allocated_mb": stats["allocated_mb"],
            "reserved_mb": stats["reserved_mb"],
            "total_mb": stats["total_mb"],
            "free_mb": stats["free_mb"],
            "peak_mb": round(self._peak_vram_mb, 1),
            "status_text": status_text,
        }

    def reset_peak_vram(self) -> None:
        """Reset the peak VRAM tracker."""
        with self._lock:
            stats = self.get_vram_stats()
            self._peak_vram_mb = stats["allocated_mb"]

    def _unload_model(self, name: str) -> None:
        """Unload a single model with thorough GPU cleanup."""
        bundle = self._active_models.pop(name, None)
        if bundle is None:
            return

        before = self.get_vram_stats()
        logger.info(
            f"Unloading model '{name}'. VRAM before unload: "
            f"{before['allocated_mb']}MB allocated / {before['reserved_mb']}MB reserved"
        )

        # 1. Custom unloader hook
        if name in self._unloaders:
            try:
                self._unloaders[name](bundle)
            except Exception as e:
                logger.warning(f"Error in unloader for {name}: {e}")

        # 2. Move sub-components to CPU if hasattr
        self._deep_move_to_cpu(bundle)

        # 3. Delete references
        del bundle

        # 4. Thorough garbage collection & CUDA cache clearing
        gc.collect()
        if self.device == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception as e:
                logger.debug(f"CUDA synchronization error during cleanup: {e}")

        after = self.get_vram_stats()
        logger.info(
            f"Model '{name}' unloaded. VRAM after unload: "
            f"{after['allocated_mb']}MB allocated / {after['reserved_mb']}MB reserved"
        )

    def _deep_move_to_cpu(self, obj: Any) -> None:
        """Recursively attempt to move torch modules or tensors to CPU."""
        if obj is None:
            return
        if hasattr(obj, "to") and callable(getattr(obj, "to")):
            try:
                obj.to("cpu")
            except Exception:
                pass
        elif hasattr(obj, "cpu") and callable(getattr(obj, "cpu")):
            try:
                obj.cpu()
            except Exception:
                pass

        if isinstance(obj, (tuple, list)):
            for item in obj:
                self._deep_move_to_cpu(item)
        elif isinstance(obj, dict):
            for v in obj.values():
                self._deep_move_to_cpu(v)

    def unload_all(self) -> None:
        """Unload every active model on GPU."""
        with self._lock:
            active = list(self._active_models.keys())
            for name in active:
                self._unload_model(name)

    def load(self, name: str, allow_coexist: bool = False) -> Any:
        """Load requested model onto device according to strict coexistence rules."""
        with self._lock:
            if name in self._active_models:
                return self._active_models[name]

            if name not in self._loaders:
                raise ValueError(f"No loader registered for model: '{name}'")

            # Enforce Coexistence Rules:
            # Rule: before loading any model, unload every other GPU-resident model.
            # Exception: at SEARCH time, "clip" and "reranker" may coexist, but never together with "florence".
            allowed_coexisting: Set[str] = set()
            if allow_coexist and name in ("clip", "reranker"):
                allowed_coexisting = {"clip", "reranker"}

            for resident_name in list(self._active_models.keys()):
                if resident_name not in allowed_coexisting:
                    logger.info(
                        f"Unloading resident model '{resident_name}' before loading '{name}' "
                        f"(coexistence not permitted)."
                    )
                    self._unload_model(resident_name)

            before = self.get_vram_stats()
            logger.info(
                f"Loading model '{name}' onto {self.device}. VRAM before load: "
                f"{before['allocated_mb']}MB allocated / {before['reserved_mb']}MB reserved"
            )

            bundle = self._loaders[name]()
            self._active_models[name] = bundle

            after = self.get_vram_stats()
            logger.info(
                f"Model '{name}' loaded. VRAM after load: "
                f"{after['allocated_mb']}MB allocated / {after['reserved_mb']}MB reserved"
            )
            return bundle

    @contextmanager
    def use(
        self,
        *model_names: str,
        allow_coexist: bool = False,
    ) -> Iterator[Union[Any, Tuple[Any, ...]]]:
        """Context manager API for model execution.
        
        Example:
            with manager.use("florence") as florence_bundle:
                ...
            # Model is automatically unloaded on exit.

            # Coexisting search context:
            with manager.use("clip", "reranker") as (clip_bundle, rerank_bundle):
                ...
        """
        if not model_names:
            raise ValueError("At least one model name must be provided to manager.use()")

        is_multi = len(model_names) > 1
        coexist_flag = allow_coexist or is_multi

        with self._lock:
            bundles = []
            for name in model_names:
                b = self.load(name, allow_coexist=coexist_flag)
                bundles.append(b)

            try:
                if is_multi:
                    yield tuple(bundles)
                else:
                    yield bundles[0]
            finally:
                # Unload loaded models in reverse order
                for name in reversed(model_names):
                    self._unload_model(name)

    @contextmanager
    def use_search(self) -> Iterator[Tuple[Any, Any]]:
        """Convenience context manager for search time: CLIP + Reranker coexisting on GPU."""
        with self.use("clip", "reranker", allow_coexist=True) as bundles:
            yield bundles


# Global singleton instance
_GLOBAL_MANAGER: Optional[ModelManager] = None
_INIT_LOCK = threading.Lock()


def get_model_manager(device: Optional[str] = None) -> ModelManager:
    """Retrieve or initialize the global thread-safe ModelManager instance."""
    global _GLOBAL_MANAGER
    with _INIT_LOCK:
        if _GLOBAL_MANAGER is None:
            _GLOBAL_MANAGER = ModelManager(device=device)
            # Register standard loaders from src.models
            _register_default_loaders(_GLOBAL_MANAGER)
        return _GLOBAL_MANAGER


def _register_default_loaders(manager: ModelManager) -> None:
    """Register CLIP, Florence-2, and Reranker loaders with manager."""
    import src.models as models

    manager.register_loader(
        "clip",
        load_fn=lambda: models.load_clip(device=manager.device),
        unload_fn=lambda bundle: models.unload_clip(bundle),
    )

    manager.register_loader(
        "florence",
        load_fn=lambda: models.load_florence(device=manager.device),
        unload_fn=lambda bundle: models.unload_florence(bundle),
    )

    manager.register_loader(
        "reranker",
        load_fn=lambda: models.load_reranker(device=manager.device),
        unload_fn=lambda bundle: models.unload_reranker(bundle),
    )
