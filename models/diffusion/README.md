# Diffusion Restoration (DDPM / DDIM)

Owner: @SachinB18

## Overview
Conditional Denoising Diffusion Probabilistic Model (DDPM) with DDIM deterministic sampling for restoring degraded PPE inspection crops (motion blur, low light, compression, sensor noise).

- **Notebook**: `diffusion_restoration.ipynb`
- **Design Review**: `docs/diffusion_design.md`
- **Trained Artifacts**: `results/diffusion/` (`diffusion_best.pt`, `regression_best.pt`, `model_config.json`, `diffusion_metrics.json`, figures)
- **Dashboard Service**: `dashboard/models/diffusion/`
- **Active Branch**: `feature/diffusion`

## Contribution Guidelines
Pull requests that touch only `/models/diffusion/` can be merged after one review from `@SachinB18`. Pull requests modifying `/common/` or `/dashboard/` require approval from `@Mohit-developer-tech`.

