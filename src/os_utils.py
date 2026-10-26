"""Operating System & Native Desktop Utilities for EmbedEye.

Provides cross-platform facilities for launching native file managers,
revealing files, and navigating directories across Windows, macOS, and Linux.
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path


class NativeExplorerLauncher:
    """Opens local directories or reveals files in the OS native file explorer."""

    @staticmethod
    def open_folder(path_str: str) -> str:
        """Open containing directory in OS native file explorer.

        Args:
            path_str: Path string of file or folder.

        Returns:
            Status message string.
        """
        if not path_str or not path_str.strip():
            return "No path provided."

        target = Path(path_str.strip())
        folder = target.parent if target.is_file() else target

        if not folder.exists():
            return f"Directory does not exist: {folder}"

        system = platform.system()
        try:
            if system == "Windows":
                if target.is_file():
                    subprocess.Popen(["explorer", f"/select,{str(target.resolve())}"])
                else:
                    os.startfile(str(folder.resolve()))
            elif system == "Darwin":
                subprocess.Popen(["open", str(folder.resolve())])
            else:
                subprocess.Popen(["xdg-open", str(folder.resolve())])
            return f"Opened folder: {folder}"
        except Exception as e:
            return f"Failed to open folder: {e}"


def open_folder(path_str: str) -> str:
    """Facade for NativeExplorerLauncher.open_folder."""
    return NativeExplorerLauncher.open_folder(path_str)
