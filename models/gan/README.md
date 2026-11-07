# Conditional GAN (cGAN) with DiffAugment

Owner: @SachinB18

## Overview
Conditional DCGAN with Spectral Normalization, Projection Discriminator, Class-Conditional BatchNorm, and DiffAugment for synthetic PPE generation and rare-class (`head`) training augmentation with privacy nearest-neighbor safeguards.

- **Notebook**: `gan_conditional.ipynb`
- **Design Review**: `docs/gan_design.md`
- **Trained Artifacts**: `results/cgan/` (`cgan_best.pt`, `model_config.json`, `cgan_metrics.json`, figures)
- **Dashboard Service**: `dashboard/models/cgan/`
- **Active Branch**: `feature/gan`

## Contribution Guidelines
Pull requests that touch only `/models/gan/` can be merged after one review from `@SachinB18`. Pull requests modifying `/common/` or `/dashboard/` require approval from `@Mohit-developer-tech`.

