"""
Deprecated entry point. Use the installed package instead::

    pip install -e .
    from world_model.wan_flow import build_render_conditioned_wan_i2v
"""

from __future__ import annotations

import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from world_model.wan_flow.model import *  # noqa: E402,F403
