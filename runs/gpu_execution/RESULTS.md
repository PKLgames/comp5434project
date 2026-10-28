# GPU runbook execution results

Code: 4e43dff7f82999b9fe0318bf11a2d7b0bc890dc7
GPU: NVIDIA GeForce RTX 3060 Laptop GPU
PyTorch: 2.6.0+cu124 / CUDA 12.4

39 tests passed. CUDA and the ordinal+FiLM smoke gate passed.
Training used all 6,104,774 training rows and 1,526,196 validation rows.
This run uses the original high-dimensional cache (55,731 / 182,382 compact vocabularies), not the reduced-dimension experiment.

| Method | Validation RMSE | MAE | Delta vs reference | 95% paired bootstrap CI | Adopt |
| --- | ---: | ---: | ---: | --- | --- |
| Reference stack (5-fold OOF) | 1.154280 | 0.872799 | 0 | - | baseline |
| gpu_ord | 1.157757 | 0.890224 | +0.003477 | [+0.003221, +0.003737] | False |
| gpu_film | 1.158153 | 0.887843 | +0.003873 | [+0.003606, +0.004143] | False |
| gpu_ord_film | 1.155885 | 0.882161 | +0.001604 | [+0.001342, +0.001859] | False |

Each comparison used 5,000 paired resamples. All candidates have higher RMSE than the reference. Keep the reference; optional phase-7 ensembles were not started.

## Training durations

| Run | Best epoch | Training-loop seconds (including validation) |
| --- | ---: | ---: |
| gpu_ref_base | 2 | 86.68 |
| gpu_ref_tmlp | 2 | 63.24 |
| gpu_ord | 1 | 95.71 |
| gpu_film | 1 | 99.72 |
| gpu_ord_film | 1 | 95.40 |

## Artifacts

- runs/gpu_ref_base/best.pt and runs/gpu_ref_tmlp/best.pt: reference members.
- runs/gpu_ref_stack/stack.json: fitted ensemble weights/intercept and OOF metrics.
- runs/gpu_ord, runs/gpu_film, runs/gpu_ord_film: candidate weights, summaries and comparisons.
- runs/gpu_execution/environment.json and requirements-frozen.txt: exact local environment.
- runs/gpu_execution/execution.json and *.log: phase status, commands and timings.
- artifacts is a directory junction to the existing original cache; no second 2.4 GB cache was created.

The original published per-row prediction file was not supplied, so vs_published is unavailable. The mandatory reference was instead reproduced on this GPU as required.

The combined-method command uses the defaults for lr/weight decay specified in the runbook block; the individual methods use explicit 0.003 / 0.0001.

No test-set predictions, Kaggle submissions or remote Git writes were performed. High-dimensional checkpoints must be paired with their original feature processors and cache vocabularies; the earlier 4,096 / 16,384-dimensional test features belong to a different model.


## Published artifact bundle

All five GPU checkpoints and six validation prediction arrays are stored with Git LFS. After cloning, use `git lfs pull` to retrieve their contents.

`cache_reference/` contains the original cache metadata and both compact vocabularies, not the training/validation shards. Paths in this metadata describe the original machine and must be adapted when rebuilding a cache.

`runbook_driver.py` archives the execution driver (relocated to this directory). It expects the cache directory as its first argument and refuses to overwrite existing training runs. `artifact_manifest.json` lists SHA-256 hashes and sizes of all files in this experiment bundle.
