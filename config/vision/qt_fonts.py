"""Use installed fonts when an OpenCV wheel points Qt at missing bundled fonts."""
import os
from pathlib import Path
import sys


def configure_qt_fonts():
    """Call after importing cv2, before creating the first OpenCV window.

    Linux OpenCV wheels overwrite QT_QPA_FONTDIR during import, even when the
    wheel contains no qt/fonts directory. Setting it before import is ineffective.
    This only changes the current process; no conda package files are modified.
    """
    if not sys.platform.startswith("linux"):
        return None
    configured = os.environ.get("QT_QPA_FONTDIR")
    candidates = ([Path(configured)] if configured else []) + [
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/truetype/liberation2"),
        Path("/usr/share/fonts/truetype/liberation"),
        Path("/usr/share/fonts/dejavu"),
        Path(sys.prefix) / "fonts",
    ]
    for directory in candidates:
        if directory.is_dir() and any(
                path.suffix.lower() in (".ttf", ".otf", ".ttc")
                for path in directory.iterdir()):
            os.environ["QT_QPA_FONTDIR"] = str(directory)
            return directory
    return None
