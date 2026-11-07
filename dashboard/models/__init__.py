"""
dashboard.models package
Registry and routers for generative PPE models.
"""

from dashboard.models.registry import (
    MODEL_REGISTRY,
    get_registered_models,
    get_model_info
)
from dashboard.models.autoencoder.router import router as autoencoder_router
from dashboard.models.cvae.router import router as cvae_router
from dashboard.models.cgan.router import router as cgan_router
from dashboard.models.diffusion.router import router as diffusion_router
from dashboard.models.policy.router import router as policy_router

__all__ = [
    "MODEL_REGISTRY",
    "get_registered_models",
    "get_model_info",
    "autoencoder_router",
    "cvae_router",
    "cgan_router",
    "diffusion_router",
    "policy_router",
]

