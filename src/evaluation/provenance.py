"""Run provenance capture.

An evaluation number is only reproducible if you know what produced it. This
module records the code version, environment, and configuration behind a run
and attaches it to the report.

Everything here degrades gracefully: the Docker image ships without a `.git`
directory and without some optional packages, so every lookup is guarded and
returns `"unavailable"` rather than raising. Provenance capture must never be
able to fail an evaluation.
"""
from __future__ import annotations

import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path

#: Packages whose versions materially affect results.
TRACKED_PACKAGES: tuple[str, ...] = (
    "chromadb",
    "numpy",
    "openai",
    "anthropic",
    "sentence-transformers",
    "tiktoken",
    "pydantic",
    "fastapi",
)

UNAVAILABLE = "unavailable"


def _run_git(args: list[str], cwd: Path) -> str:
    """Run a git command, returning UNAVAILABLE on any failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return UNAVAILABLE
    if result.returncode != 0:
        return UNAVAILABLE
    return result.stdout.strip() or UNAVAILABLE


def git_info(repo_root: Path | None = None) -> dict[str, str | bool]:
    """Current commit, branch and dirty flag, or UNAVAILABLE outside a checkout."""
    root = repo_root or Path(__file__).resolve().parents[2]
    commit = _run_git(["rev-parse", "HEAD"], root)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    status = _run_git(["status", "--porcelain"], root)
    return {
        "commit": commit,
        "branch": branch,
        "dirty": status not in (UNAVAILABLE, ""),
    }


def package_versions() -> dict[str, str]:
    """Installed versions of the packages that can change results."""
    versions: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = pkg_version(name)
        except PackageNotFoundError:
            versions[name] = UNAVAILABLE
    return versions


def collect_provenance(**extra: object) -> dict:
    """Assemble the provenance block for a run.

    Any keyword arguments are merged in, which is how callers attach the
    dataset path, configuration fingerprints and pipeline identity.
    """
    provenance: dict = {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_info(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
    }
    provenance.update(extra)
    return provenance


def describe_component(obj: object) -> str:
    """Stable, human-readable identity for a pipeline component.

    Records the class name plus the model identifier when the component exposes
    one, so a report says `OpenAILLM(gpt-4o-mini)` rather than just `object`.
    """
    if obj is None:
        return UNAVAILABLE
    name = type(obj).__name__
    for attr in ("model", "model_name"):
        value = getattr(obj, attr, None)
        if isinstance(value, str) and value:
            return f"{name}({value})"
        if value is not None and hasattr(value, "__class__") and attr == "model":
            inner = getattr(value, "name_or_path", None)
            if isinstance(inner, str) and inner:
                return f"{name}({inner})"
    return name
