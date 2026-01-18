"""sca_run: minimal runtime for streaming audio -> Qwen3-Omni (Transformers backend).

This package is intentionally small and config-driven.
"""

from .config import AppConfig, load_config

__all__ = ["AppConfig", "load_config"]
