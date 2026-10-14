from pathlib import Path
import os


def find_project_root() -> Path:
    """
    Returns the project root: the folder that contains `settings/`.
    Taken from the location of this file, so the project is found wherever a script is
    launched from.
    """
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = find_project_root()


def resolve_path(user_path: str | Path) -> Path:
    """
    Converts user path to absolute path.

    Supports:
    - absolute paths
    - ~/ paths
    - relative paths (relative to PROJECT_ROOT)
    """

    p = Path(os.path.expandvars(str(user_path))).expanduser()

    if p.is_absolute():
        return p.resolve()

    return (PROJECT_ROOT / p).resolve()
