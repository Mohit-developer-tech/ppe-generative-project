# Publishing Shared Assets to GitHub Releases (Tag: `v0.1`)

This guide provides the exact steps for packaging and publishing the shared project assets (`release_assets/`) to a private or public GitHub Release tagged `v0.1`.

---

## 1. Prerequisites
- **GitHub CLI (`gh`)**: Ensure `gh` is installed and authenticated:
  ```bash
  gh auth status
  ```
  *(If not authenticated, run `gh auth login` and follow the prompts).*
- **Shared Assets Manifest**: Ensure `assets_manifest.json` matches the contents in `release_assets/`:
  ```bash
  python scripts/prepare_release_assets.py
  ```

---

## 2. Release Contents Inventory
Only the shared baseline assets are included in `v0.1` (experimental sanity runs of cVAE, cGAN, Diffusion, and Policy models are excluded from initial release):

| Rel Path in `release_assets/` | Target Repo Path | Description |
|---|---|---|
| `autoencoder/ae_best.pt` | `artifacts/autoencoder/ae_best.pt` | ConvAE baseline weights |
| `autoencoder/scaler.joblib` | `artifacts/autoencoder/scaler.joblib` | StandardScaler for 256-d latent code |
| `autoencoder/svm_model.joblib` | `artifacts/autoencoder/svm_model.joblib` | Downstream linear SVM classifier |
| `autoencoder/latent_pca2d.joblib` | `artifacts/autoencoder/latent_pca2d.joblib` | 2D PCA projection transformer |
| `autoencoder/latent_features.npz` | `artifacts/autoencoder/latent_features.npz` | Cached latent features for quick evaluation |
| `autoencoder_v2/ae_best.pt` | `artifacts/autoencoder_v2/v2_loss_combined_256/ae_best.pt` | V2 combined-loss Autoencoder |
| `judge/judge_seed_42.pt` | `artifacts/judge/judge_seed_42.pt` | SmallResNet Judge (Seed 42) |
| `judge/judge_seed_43.pt` | `artifacts/judge/judge_seed_43.pt` | SmallResNet Judge (Seed 43) |
| `judge/judge_seed_44.pt` | `artifacts/judge/judge_seed_44.pt` | SmallResNet Judge (Seed 44) |
| `judge/judge_seed_45.pt` | `artifacts/judge/judge_seed_45.pt` | SmallResNet Judge (Seed 45) |
| `judge/judge_seed_46.pt` | `artifacts/judge/judge_seed_46.pt` | SmallResNet Judge (Seed 46) |
| `noise_floor.json` | `results/noise_floor.json` | Empirical FID reference noise floor |

---

## 3. Creating & Uploading the Release via GitHub CLI

### Option A: Create Tag & Release in One Command
Run the following from the project root:

```bash
gh release create v0.1 \
  --repo Mohit-developer-tech/ppe-generative-project \
  --title "v0.1 - Shared Baseline & Judge Ensemble Assets" \
  --notes "Shared baseline weights, 5-seed ensemble judge, and reference FID noise floor for the PPE Generative AI Benchmark suite." \
  release_assets/autoencoder/* \
  release_assets/autoencoder_v2/* \
  release_assets/judge/* \
  release_assets/noise_floor.json \
  assets_manifest.json
```

### Option B: Upload to an Existing Tagged Release
If the release draft or tag `v0.1` already exists:

```bash
gh release upload v0.1 \
  --repo Mohit-developer-tech/ppe-generative-project \
  release_assets/autoencoder/* \
  release_assets/autoencoder_v2/* \
  release_assets/judge/* \
  release_assets/noise_floor.json \
  assets_manifest.json \
  --clobber
```

---

## 4. Verification After Publishing
Verify that team members or clean clones can pull and verify the release assets:

```bash
python scripts/fetch_assets.py --from-gh --tag v0.1 --repo Mohit-developer-tech/ppe-generative-project
```
Every file will be checksum-verified against `assets_manifest.json`.

