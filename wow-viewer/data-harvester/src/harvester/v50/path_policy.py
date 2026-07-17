"""Approved-root/protected-root path resolution that rejects symlink escapes (Spec 109 T011,
FR-020/FR-021).

Every check here resolves the candidate path fully (following any symlinks) via
``Path.resolve(strict=True)`` before comparing it against configured roots. That single resolve
call is what catches an escape: if a symlink under an approved root points outside every approved
root, the *resolved* path is what gets compared, not the nominal one, so the redirection cannot
sneak a path through. Protected roots always win over approved roots -- a path can never be treated
as safe-to-touch just because some approved root also happens to contain it.
"""

from __future__ import annotations

from pathlib import Path


class PathPolicyError(ValueError):
    """Raised when a path fails approved-root/protected-root policy."""


class PathPolicy:
    def __init__(self, *, approved_roots: list[Path] | tuple[Path, ...], protected_roots: list[Path] | tuple[Path, ...]):
        self._approved_roots = tuple(Path(root).resolve(strict=True) for root in approved_roots)
        self._protected_roots = tuple(Path(root).resolve(strict=True) for root in protected_roots)
        if not self._approved_roots:
            raise ValueError("at least one approved root is required")

    @property
    def approved_roots(self) -> tuple[Path, ...]:
        return self._approved_roots

    @property
    def protected_roots(self) -> tuple[Path, ...]:
        return self._protected_roots

    @staticmethod
    def _is_within(candidate: Path, root: Path) -> bool:
        return candidate == root or root in candidate.parents

    def is_protected(self, path: Path) -> bool:
        resolved = Path(path).resolve(strict=True)
        return any(self._is_within(resolved, root) for root in self._protected_roots)

    def is_approved(self, path: Path) -> bool:
        resolved = Path(path).resolve(strict=True)
        if any(self._is_within(resolved, root) for root in self._protected_roots):
            return False
        return any(self._is_within(resolved, root) for root in self._approved_roots)

    def resolve_within_approved_root(self, path: Path) -> Path:
        """Resolve ``path`` and return it only if it is inside an approved root and not inside
        any protected root (even a symlink escape). Raises ``PathPolicyError`` otherwise."""
        try:
            resolved = Path(path).resolve(strict=True)
        except OSError as exc:
            raise PathPolicyError(f"path does not exist or cannot be resolved: {path}") from exc

        if any(self._is_within(resolved, root) for root in self._protected_roots):
            raise PathPolicyError(f"path resolves into a protected root, refusing: {path} -> {resolved}")

        if not any(self._is_within(resolved, root) for root in self._approved_roots):
            raise PathPolicyError(
                f"path does not resolve inside any approved root, refusing: {path} -> {resolved}"
            )

        return resolved
