# Policy-Driven Compliance Checker

Owner: @KhOmkar

## Overview
NLP & Multi-modal compliance verification pipeline that reads safety regulations (OSHA 29 CFR 1926, MUTCD, ANSI/ISEA 107, USACE EM 385-1-1), classifies required PPE clauses via fine-tuned and temperature-calibrated DistilBERT, and executes a rigorous 3-state compliance verdict (`Compliant`, `Violation`, `Cannot determine`).

- **Notebook**: `policy_checker.ipynb`
- **Design Review**: `docs/policy_design.md`
- **Dataset**: `data/policy/` (`labels.csv`, `evaluation_slices.csv`, `README.md`)
- **Trained Artifacts**: `results/policy/` (`policy_classifier.pt`, `model_config.json`, `policy_metrics.json`, figures)
- **Dashboard Service**: `dashboard/models/policy/`
- **Active Branch**: `feature/policy`

## Contribution Guidelines
Pull requests that touch only `/models/policy/` can be merged after one review from `@KhOmkar`. Pull requests modifying `/common/` or `/dashboard/` require approval from `@Mohit-developer-tech`.

