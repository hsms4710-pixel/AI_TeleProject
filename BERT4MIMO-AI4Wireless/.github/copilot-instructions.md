# BERT4MIMO AI Coding Agent Instructions

## Project Overview

BERT4MIMO is a BERT-based framework for Channel State Information (CSI) processing in massive MIMO wireless communications. The project applies Transformer encoders to predict, compress, and enhance CSI for multi-cell, multi-user, multi-antenna scenarios.

**Core Value**: Reduces pilot overhead by 60%, achieves 10:1 to 50:1 compression ratios, improves denoising accuracy by 50%, and boosts system capacity by 20%+.

## Architecture & Data Flow

### Three-Component Design

1. **Model Layer** (`model.py`, `csibert_util.py`)
   - `CSIBERT` class wraps HuggingFace BertModel with custom embeddings
   - Time + feature embeddings combined before passing to BERT encoder
   - Input: `(batch, seq_len, feature_dim)` where `feature_dim = Tx × Rx × 2` (real/imag CSI components)
   - Output: Per-token predictions of shape `(batch, seq_len, feature_dim)`
   - **Critical**: `feature_dim` must match checkpoint; models are NOT transferable across different antenna configurations

2. **Training Pipeline** (`train.py`)
   - Loads `.mat` files from `foundation_model_data/` (MATLAB-generated CSI)
   - CSI preprocessing: normalize real/imag separately → flatten `(time, subcarriers, Tx, Rx, 2)` to `(time, feature_dim)`
   - **Masking strategy**: 15% of time steps zeroed out (not token-level masking)
   - Saves checkpoints to `checkpoints/` with full config (hidden_size, num_layers, feature_dim, etc.)

3. **WebUI** (`webui/app.py`)
   - Gradio interface with 4 tabs: Training, Import Data, Data Generation, Experiments
   - **Auto-loads latest model** on startup by scanning `checkpoints/*.pt` by modification time
   - `TrainingManager` class handles model lifecycle and experiment execution
   - Calls MATLAB via subprocess for data generation (`data_generator.m`)

### Data Generation (MATLAB Required)

- `data_generator.m` simulates 5G NR TDL channels with configurable parameters:
  - `numCells`, `numUEs`, `numSubcarriers`, `massiveMIMONumAntennas`, `numReceiveAntennas`
  - Generates 3 scenarios per UE: stationary (TDL-A), low mobility (TDL-B), high mobility (TDL-C)
- Output: `.mat` file with nested cell structure `multi_cell_csi{cell, ue}[scenarios]`
- **Python cannot parse this directly** - preprocessing in `train.py` iterates nested cells

## Development Workflows

### Quick Start (Windows)

```powershell
RUN.bat  # Auto-creates venv, installs deps, launches WebUI at http://127.0.0.1:7861
```

**Linux/Mac**: Use `bash RUN.sh` instead. Both scripts:
1. Detect or create `.venv` 
2. Install PyTorch (CUDA 11.8 index by default) + requirements
3. Launch WebUI with `python webui/app.py`

### Training Configurations

Three presets optimized for different hardware:

| Config | Hidden Size | Layers | Heads | Params | GPU Memory | Training Time |
|--------|-------------|--------|-------|--------|------------|---------------|
| Lightweight | 256 | 4 | 4 | ~2.8M | 4GB | 30 min |
| Standard | 512 | 8 | 8 | ~11.2M | 8GB | 2 hours |
| Original | 768 | 12 | 12 | ~25.3M | 16GB | 8 hours |

**Batch sizes**: 16/32/64 respectively. **Epochs**: 10/50/200.

### Running Experiments

#### Via WebUI (Recommended)
- **Experiments tab** auto-detects trained models in dropdown
- 5 basic tests (`model_validation.py`): reconstruction error, prediction accuracy, SNR robustness, compression ratio, inference speed
- 8 advanced experiments (`experiments_extended.py`): masking sensitivity, scenario analysis, subcarrier performance, Doppler robustness, cross-scenario generalization, baseline comparison, error distribution, attention visualization
- Generates plots to `imgs/` and reports to `validation_results/`

#### Command Line
```powershell
# Basic validation (5 tests)
python model_validation.py

# Advanced experiments (all 8)
python run_all_experiments.py --mode all

# Single experiment
python run_all_experiments.py --mode single --experiment 3
```

### Device Priority

Code auto-detects: **CUDA > MPS (Apple Silicon) > CPU**

```python
# Pattern used throughout codebase
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

## Project-Specific Patterns

### 1. Checkpoint Structure

All `*.pt` checkpoints MUST contain:
```python
{
    'model_state_dict': ...,
    'feature_dim': int,  # CRITICAL - derived from antenna config
    'hidden_size': int,
    'num_hidden_layers': int,
    'num_attention_heads': int,
    # Training metadata (optional but recommended)
    'epoch': int,
    'train_loss': float
}
```

**When loading models**: Extract `feature_dim` FIRST to initialize `CSIBERT` correctly. See `webui/app.py:auto_load_model()` for reference.

### 2. CSI Data Format

- **MATLAB output**: `(time, subcarriers, Tx, Rx)` complex-valued
- **After preprocessing**: `(batch, time, feature_dim)` where `feature_dim = subcarriers × Tx × Rx × 2`
- **Masking**: Applied at time-step level, not individual features

### 3. WebUI Model Management

`TrainingManager.scan_available_models()` builds dropdown list:
- Scans `checkpoints/*.pt`
- Extracts config from each checkpoint
- Sorts by modification time (newest first)
- Display format: `"filename.pt (H:512, L:8, A:8)"`

**When adding model features**: Update this method to include new config parameters in display string.

### 4. Experiment Results Storage

- Plots: `imgs/*.png` (overwritten each run)
- Reports: `validation_results/validation_report.json` (structured metrics)
- Markdown summary: `validation_results/VALIDATION_REPORT.md` (human-readable)

**Convention**: Each experiment function returns `(plot_paths, metrics_dict)`. WebUI collects these into combined report.

### 5. Python-MATLAB Bridge

WebUI's `run_data_generation()` uses subprocess:
```python
cmd = f'matlab -batch "numCells={cells}; ... data_generator"'
subprocess.run(cmd, shell=True)
```

**Error handling**: Check `stdout/stderr` for MATLAB errors. Common issue: MATLAB not in PATH on Windows.

## Integration Points

### HuggingFace Transformers
- Uses `BertConfig` and `BertModel` from `transformers` library
- **Custom embeddings** replace default token embeddings
- Access attention weights via `output_attentions=True` in forward pass

### PyTorch Training Loop
- Optimizer: `AdamW` (PyTorch native, not HF's `transformers.AdamW`)
- Scheduler: `transformers.get_scheduler()` with linear warmup
- No gradient clipping applied (add if training instability occurs)

### Gradio UI
- All training runs in background threads to avoid blocking UI
- Use `gr.Progress()` wrapper for long-running tasks
- **Critical**: Model training updates status via `TrainingManager.log_status()` - check this method when debugging UI feedback

## Common Pitfalls

1. **Feature Dimension Mismatch**: Always verify `checkpoint['feature_dim']` matches your data's `Tx × Rx × 2`. Symptoms: shape errors in forward pass.

2. **MATLAB Data Loading**: Nested cell arrays require careful iteration. See `train.py:load_csi_data()` - don't use `scipy.io.loadmat()['multi_cell_csi'][0,0]` directly.

3. **MPS Device Limitations**: Some ops unsupported on Apple Silicon. If errors occur, add `.cpu()` temporarily or use CPU device.

4. **Path Handling**: WebUI uses `Path(__file__).parent.parent` for project root. All file paths should be absolute or relative to `PROJECT_ROOT`.

5. **Virtual Environment**: `RUN.bat` creates `.venv` in project root. If running scripts manually, activate first: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Linux/Mac).

## Documentation References

- **Architecture details**: `docs/FILES.md`
- **Usage examples**: `docs/USAGE.md` 
- **Test descriptions**: `docs/TESTS.md`
- **Future roadmap**: `FuturePlan.md` (realtime prediction, model compression, deployment strategies)

## Key Files Map

| Component | Primary File | Key Classes/Functions |
|-----------|-------------|----------------------|
| Model definition | `model.py` | `CSIBERT`, `tokenize_csi_matrix`, `collate_fn` |
| Training | `train.py` | `load_csi_data()`, `preprocess_csi_matrix()`, `mask_data()` |
| Validation | `model_validation.py` | `CSIBERTValidator` |
| Advanced tests | `experiments_extended.py` | `AdvancedCSIBERTExperiments` |
| WebUI | `webui/app.py` | `TrainingManager`, `create_ui()` |
| Data generation | `data_generator.m` | MATLAB script (channel simulation) |

## When Adding New Features

- **New model architectures**: Extend `CSIBERT` in `model.py`, update checkpoint save/load in `train.py`
- **New experiments**: Add methods to `AdvancedCSIBERTExperiments` in `experiments_extended.py`
- **WebUI enhancements**: Modify `TrainingManager` in `webui/app.py`, add new Gradio tabs in `create_ui()`
- **Data preprocessing**: Update `preprocess_csi_matrix()` in `train.py`, ensure backward compatibility with existing checkpoints
