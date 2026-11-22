#!/usr/bin/env python3
"""jas_paths.py - the ONE place that decides where Jas keeps its files.

Everything Jas writes lives INSIDE the Jas folder, in .jas/ next to this file:

    <jas folder>\\.jas\\memory.db       long-term notes      (jas_memory.py)
    <jas folder>\\.jas\\status.json     live status          (jas.py writes, jas_monitor.py reads)
    <jas folder>\\.jas\\backups\\        backups of files Jas edits (jas.py)

Override the location with the JAS_HOME environment variable (e.g. JAS_HOME=D:\\jas-data).

Older versions used %USERPROFILE%\\.jas. The first time a module calls ensure(), anything found there is COPIED
here (the old folder is left untouched), so notes and settings are not lost.
"""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # the folder jas.py, jas_ui.py, ... live in


def _choose() -> Path:
    env = os.environ.get("JAS_HOME")
    if env:
        return Path(os.path.expandvars(env)).expanduser().resolve()
    return ROOT / ".jas"


DATA = _choose()
LEGACY = Path.home() / ".jas"                   # where earlier versions put everything


def ensure() -> list:
    """Create DATA and, on first use, copy an older ~/.jas into it. Returns the names copied."""
    try:
        DATA.mkdir(parents=True, exist_ok=True)
    except OSError:
        return []
    try:
        if LEGACY.resolve() == DATA.resolve() or not LEGACY.is_dir():
            return []
        if any(DATA.iterdir()):
            return []                            # already have data here: never overwrite
        copied = []
        for item in LEGACY.iterdir():
            dest = DATA / item.name
            if dest.exists():
                continue
            try:
                shutil.copytree(item, dest) if item.is_dir() else shutil.copy2(item, dest)
                copied.append(item.name)
            except OSError:
                continue
        return copied
    except (OSError, RuntimeError):
        return []


def file(name: str) -> Path:
    """A path inside the data folder (the folder is created if needed)."""
    ensure()
    return DATA / name
