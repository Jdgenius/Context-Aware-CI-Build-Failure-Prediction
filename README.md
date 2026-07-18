# Context-Aware CI Build Failure Prediction

This project preprocesses TravisTorrent/GitHub build data into CodeBERT embedding shards, then trains two models for CI build success/failure prediction:

- A baseline `RandomForestClassifier` over concatenated CodeBERT embeddings.
- A neural attention-fusion model that learns how to weight commit-message, diff, and code-context embeddings.

## Prerequisites

Install these before running the project:

- Python `>=3.14,<3.15`
- Poetry
- Git on `PATH`
- Internet access for GitHub repository fetches and Hugging Face/PyTorch package downloads
- Optional: NVIDIA GPU driver for CUDA acceleration

On Windows, enable long paths before installing CUDA PyTorch wheels. Run PowerShell as Administrator:

```powershell
New-ItemProperty `
  -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" `
  -Value 1 `
  -PropertyType DWORD `
  -Force
```

Restart the terminal afterward. If installs still fail with `WinError 206`, restart Windows.

## Install

From the repository root:

```powershell
poetry env use 3.14
poetry install
```

The current `pyproject.toml` pins PyTorch to the CUDA 13.0 wheel source:

```toml
torch == 2.12.1+cu130
```

A CUDA PyTorch build can still run on CPU when commands use `--device cpu`, but installing it may require the Windows long-path fix above. If Poetry scripts such as `preprocess-travistorrent`, `count-build-labels`, or `count-repos` are not recognized, run `poetry install` again to refresh entry points, or use the `python -m ...` forms shown below.

Verify the environment:

```powershell
poetry run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
poetry run git --version
```

## Preprocessing

The preprocessing pipeline reads TravisTorrent rows, clones/fetches GitHub commits, extracts commit messages, diffs, and code context, embeds each text field with CodeBERT, and writes paired shard files:

```text
embedding_shards/
  manifest.json
  shard_00000.pt
  shard_00000.text.jsonl.gz
  ...
```

Run an environment check:

```powershell
poetry run preprocess-travistorrent environment-check `
  --travistorrent-csv-path src/context_aware_ci_build_failure_prediction/preprocessing/final-2017-01-25.csv `
  --output-dir embedding_shards `
  --temp-repo-root temp_repos
```

JSON environment check:

```powershell
poetry run preprocess-travistorrent environment-check `
  --travistorrent-csv-path src/context_aware_ci_build_failure_prediction/preprocessing/final-2017-01-25.csv `
  --json
```

Run preprocessing:

```powershell
poetry run preprocess-travistorrent run `
  --travistorrent-csv-path src/context_aware_ci_build_failure_prediction/preprocessing/final-2017-01-25.csv `
  --output-dir embedding_shards `
  --temp-repo-root temp_repos `
  --failure-log-path logs/failures.jsonl `
  --run-summary-path logs/preprocessing_summary.json `
  --overwrite
```

Use GPU if available:

```powershell
poetry run preprocess-travistorrent run `
  --travistorrent-csv-path src/context_aware_ci_build_failure_prediction/preprocessing/final-2017-01-25.csv `
  --output-dir embedding_shards `
  --temp-repo-root temp_repos `
  --failure-log-path logs/failures.jsonl `
  --run-summary-path logs/preprocessing_summary.json `
  --device cuda `
  --overwrite
```

Force CPU:

```powershell
poetry run preprocess-travistorrent run `
  --travistorrent-csv-path src/context_aware_ci_build_failure_prediction/preprocessing/final-2017-01-25.csv `
  --output-dir embedding_shards `
  --temp-repo-root temp_repos `
  --failure-log-path logs/failures.jsonl `
  --run-summary-path logs/preprocessing_summary.json `
  --device cpu `
  --overwrite
```

Resume an interrupted run:

```powershell
poetry run preprocess-travistorrent run `
  --travistorrent-csv-path src/context_aware_ci_build_failure_prediction/preprocessing/final-2017-01-25.csv `
  --output-dir embedding_shards `
  --temp-repo-root temp_repos `
  --failure-log-path logs/failures.jsonl `
  --run-summary-path logs/preprocessing_summary.json `
  --resume
```

Useful preprocessing flags:

- `--max-repos N`: limit the number of repos processed.
- `--shard-size N`: samples per shard, default `5000`.
- `--raw-batch-size N`: raw samples collected before embedding, default `64`.
- `--embed-batch-size N`: CodeBERT embedding batch size, default `32`.
- `--repo-timing-log-path logs/repo_timing.jsonl`: write per-repo timing records.

Equivalent module form:

```powershell
poetry run python -m context_aware_ci_build_failure_prediction.preprocessing.cli run ...
```

## Dataset Info Commands

Count successful and unsuccessful build labels in an embedding shard directory:

```powershell
poetry run python -m context_aware_ci_build_failure_prediction.info.label_counts `
  --source-dir embedding_shards
```

Save label counts:

```powershell
poetry run python -m context_aware_ci_build_failure_prediction.info.label_counts `
  --source-dir embedding_shards `
  --output-path logs/build_label_counts.json
```

Count unique repos and samples per repo:

```powershell
poetry run python -m context_aware_ci_build_failure_prediction.info.repo_counts `
  --source-dir embedding_shards
```

Save repo counts:

```powershell
poetry run python -m context_aware_ci_build_failure_prediction.info.repo_counts `
  --source-dir embedding_shards `
  --output-path logs/repo_counts.json
```

After `poetry install`, these shorter commands should also work:

```powershell
poetry run count-build-labels --source-dir embedding_shards
poetry run count-repos --source-dir embedding_shards
```

## Train Attention Fusion Model

Train the neural attention-fusion model:

```powershell
poetry run python -m context_aware_ci_build_failure_prediction.models.attention_fusion.train `
  --source-dir embedding_shards `
  --checkpoint-path checkpoints/attention_fusion.pt `
  --num-samples 10000 `
  --epochs 30 `
  --batch-size 32 `
  --learning-rate 0.001 `
  --validation-fraction 0.2 `
  --test-fraction 0.2 `
  --seed 0 `
  --device cuda `
  --curve-path logs/attention_fusion_curves.png `
  --confusion-matrix-path logs/attention_fusion_confusion_matrix.png `
  --no-show-curves `
  --no-show-confusion-matrix `
  --summary-path logs/attention_fusion_train_summary.json
```

Force CPU training:

```powershell
poetry run python -m context_aware_ci_build_failure_prediction.models.attention_fusion.train `
  --source-dir embedding_shards `
  --checkpoint-path checkpoints/attention_fusion.pt `
  --num-samples 10000 `
  --device cpu `
  --no-show-curves `
  --no-show-confusion-matrix
```

Run attention-fusion inference:

```powershell
poetry run python -m context_aware_ci_build_failure_prediction.models.attention_fusion.inference `
  --checkpoint-path checkpoints/attention_fusion.pt `
  --source-dir embedding_shards `
  --num-samples 300 `
  --batch-size 32 `
  --device cuda `
  --output-path logs/attention_fusion_predictions.json
```

Useful attention model flags:

- `--model-dim`: projection dimension, default `128`.
- `--attention-dim`: attention hidden dimension, default `64`.
- `--classifier-hidden-dim`: classifier hidden dimension, default `128`.
- `--dropout`: dropout rate, default `0.2`.
- `--shared-projection`: use one projection network for all modalities instead of separate projections.
- `--no-shuffle-splits`: disable deterministic repo shuffling.

## Train Random Forest Baseline

The model command runner supports:

```powershell
poetry run python -m context_aware_ci_build_failure_prediction.models.run --help
```

Train the random forest baseline:

```powershell
poetry run python -m context_aware_ci_build_failure_prediction.models.run baseline-train `
  --source-dir embedding_shards `
  --model-path checkpoints/random_forest.pkl `
  --num-samples 10000 `
  --validation-fraction 0.2 `
  --test-fraction 0.2 `
  --seed 0 `
  --n-estimators 300 `
  --class-weight balanced `
  --table-path logs/random_forest_test_table.csv `
  --bar-graph-path logs/random_forest_test_accuracy.png `
  --confusion-matrix-path logs/random_forest_confusion_matrix.png `
  --no-show-bar-graph `
  --no-show-confusion-matrix `
  --summary-path logs/random_forest_train_summary.json
```

Run random forest inference/evaluation from a saved checkpoint:

```powershell
poetry run python -m context_aware_ci_build_failure_prediction.models.run baseline-infer `
  --source-dir embedding_shards `
  --model-path checkpoints/random_forest.pkl `
  --table-path logs/random_forest_inference_table.csv `
  --bar-graph-path logs/random_forest_inference_accuracy.png `
  --no-show-bar-graph `
  --summary-path logs/random_forest_inference_summary.json
```

Run the random forest sample-size sweep:

```powershell
poetry run python -m context_aware_ci_build_failure_prediction.models.run baseline-sample-sweep `
  --source-dir embedding_shards `
  --table-path logs/random_forest_sample_sweep.csv `
  --bar-graph-path logs/random_forest_sample_sweep_accuracy.png `
  --summary-path logs/random_forest_sample_sweep_summary.json `
  --no-show-bar-graph
```

Default sweep sample sizes:

```text
1000, 2500, 5000, 7500, 10000
```

Override them:

```powershell
poetry run python -m context_aware_ci_build_failure_prediction.models.run baseline-sample-sweep `
  --source-dir embedding_shards `
  --sample-sizes 1000,2500,5000,7500,10000 `
  --no-show-bar-graph
```

## Splitting Behavior

Model loading uses repo-aware splits. Samples from the same repo are kept in the same set. The splitter tries to balance sample counts across train/validation/test while preserving repo boundaries.

Defaults:

- validation fraction: `0.2`
- test fraction: `0.2`
- seed: `0`
- shuffle splits: enabled

With very few repos or highly imbalanced repo sizes, exact sample percentages may be impossible. For example, if one repo dominates the loaded sample budget, that entire repo must stay in one split.

## Tests

Run tests with:

```powershell
poetry run pytest
```

Run one test file:

```powershell
poetry run pytest tests/test_preprocessing_resume.py
```

## Common Issues

### Command is not recognized

If a Poetry script is not recognized, refresh entry points:

```powershell
poetry install
```

Or use the module command:

```powershell
poetry run python -m context_aware_ci_build_failure_prediction.preprocessing.cli --help
```

### CUDA is not visible

Check PyTorch:

```powershell
poetry run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

If `torch.cuda.is_available()` is `False`, verify that your NVIDIA driver is installed and that the installed PyTorch wheel matches your CUDA setup.

### Torch install fails with long path errors on Windows

Enable Windows long paths as shown in the prerequisites section. You can also shorten Poetry virtualenv paths:

```powershell
poetry config virtualenvs.path C:\pvenvs
poetry env remove --all
poetry install
```

### Existing preprocessing outputs block a run

Use `--overwrite` to remove generated preprocessing outputs, or `--resume` to continue from a compatible existing manifest. Do not use both flags together.