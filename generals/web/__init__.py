"""Browser-based rendering support for Generals games."""

from .schemas import build_snapshot, serialize_policy_preview
from .session import WebGameSession, WebSessionConfig

__all__ = ["WebGameSession", "WebSessionConfig", "build_snapshot", "serialize_policy_preview"]
