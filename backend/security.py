"""
Security and validation module for EmbedEye backend.
Provides bearer token authentication, origin filtering, and folder path sanitization.
"""

import os
import sys
from pathlib import Path
from typing import Optional
from fastapi import Header, HTTPException, Request, status

EMBEDEYE_TOKEN_ENV = "EMBEDEYE_TOKEN"


def get_expected_token() -> Optional[str]:
    """Return the expected auth token configured in environment."""
    return os.environ.get(EMBEDEYE_TOKEN_ENV, "").strip() or None


def verify_auth_token(authorization: Optional[str] = Header(None)) -> bool:
    """
    Validate incoming Authorization header against EMBEDEYE_TOKEN.
    Raises HTTPException(401) on missing or mismatched token.
    """
    expected = get_expected_token()
    if not expected:
        # If no token configured in environment, allow local development
        return True

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts = authorization.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization scheme. Expected 'Bearer <token>'",
            headers={"WWW-Authenticate": "Bearer"},
        )

    provided = parts[1].strip()
    if provided != expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Invalid authentication token",
        )
    return True


def check_allowed_origin(request: Request) -> bool:
    """
    Ensure requests originate only from local processes or trusted origins.
    Rejects foreign web browser Origins to prevent Cross-Site WebSocket/HTTP Hijacking.
    """
    origin = request.headers.get("origin")
    if not origin:
        # Desktop applications (Avalonia/HttpClient) or curl do not send Origin
        return True

    origin_lower = origin.lower()
    allowed_prefixes = (
        "http://127.0.0.1",
        "https://127.0.0.1",
        "http://localhost",
        "https://localhost",
        "app://",
    )
    if any(origin_lower.startswith(prefix) for prefix in allowed_prefixes):
        return True

    return False


def normalize_and_validate_folder(folder_path: str) -> Path:
    """
    Validate and normalize a folder path:
    - Auto-trim surrounding quotes ("Copy as path")
    - Expand environment variables (e.g. %USERPROFILE%)
    - Normalize slashes and resolve to absolute Path
    - Verify path exists and is a directory
    - Works with unicode, spaces, UNC paths
    """
    if not folder_path or not folder_path.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Folder path cannot be empty",
        )

    raw = folder_path.strip()
    # Strip quotes if copied as path
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        raw = raw[1:-1].strip()

    # Expand environment variables
    expanded = os.path.expandvars(os.path.expanduser(raw))
    path_obj = Path(expanded).resolve()

    if not path_obj.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Folder does not exist: {path_obj}",
        )

    if not path_obj.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Path is not a directory: {path_obj}",
        )

    return path_obj


def paths_match(path_a: str, path_b: str) -> bool:
    """
    Compare two paths, case-insensitively on Windows.
    """
    norm_a = os.path.normpath(path_a)
    norm_b = os.path.normpath(path_b)
    if sys.platform.startswith("win"):
        return norm_a.lower() == norm_b.lower()
    return norm_a == norm_b


def is_subpath(child_path: str, parent_folder: str) -> bool:
    """
    Check if child_path is inside or equal to parent_folder (recursive).
    Handles Windows case-insensitivity and path normalization.
    """
    try:
        norm_child = Path(os.path.normpath(child_path)).resolve()
        norm_parent = Path(os.path.normpath(parent_folder)).resolve()

        if sys.platform.startswith("win"):
            child_parts = [p.lower() for p in norm_child.parts]
            parent_parts = [p.lower() for p in norm_parent.parts]
            return len(child_parts) >= len(parent_parts) and child_parts[:len(parent_parts)] == parent_parts
        else:
            return norm_child == norm_parent or norm_parent in norm_child.parents
    except Exception:
        return False
