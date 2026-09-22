"""Desktop listener package for the animatronic skeleton."""

from typing import Any

__all__ = ["AudioWorker", "launch_gui"]


def __getattr__(name: str) -> Any:
    if name == "AudioWorker":
        from skeleton.desktop.app import AudioWorker

        return AudioWorker
    if name == "launch_gui":
        from skeleton.desktop.app import launch_gui

        return launch_gui
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
