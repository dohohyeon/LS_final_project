from pathlib import Path
from typing import Any, Dict, Optional

import streamlit.components.v1 as components

_component_name = "flip_card_control"
_component_dir = Path(__file__).parent / "frontend"

_flip_card = components.declare_component(
    _component_name,
    path=str(_component_dir),
)


def flip_card_control(
    *,
    running: bool,
    speed: float,
    speed_delay: float = 1.0,
    key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Render the flip-card control component and return emitted events."""
    return _flip_card(
        running=running,
        speed=speed,
        speed_delay=speed_delay,
        key=key,
        default=None,
    )


__all__ = ["flip_card_control"]

