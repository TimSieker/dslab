"""Model registry. Importing this package registers every model module listed below."""

from . import climatology, deepesd, linear, unet  # noqa: F401  (registration side effects)
from .base import REGISTRY, Downscaler, build_model, register

__all__ = ["REGISTRY", "Downscaler", "build_model", "register"]
