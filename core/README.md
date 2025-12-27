# Core Module Documentation

This directory contains the core scripts for OCR model training, evaluation, and hyperparameter optimization.

## Table of Contents

- [Training Scripts](#training-scripts)
- [Hyperparameter Search](#hyperparameter-search)
- [Evaluation Scripts](#evaluation-scripts)
- [Model Management](#model-management)
- [Data Processing](#data-processing)
- [Analysis Tools](#analysis-tools)

---

## Training Scripts

### sft_gemma.py

Main training script for supervised fine-tuning of Gemma vision-language models for OCR tasks.

**Key Features:**
- Vision-language model fine-tuning with LoRA (Low-Rank Adaptation)
- Support for 4-bit quantization to reduce memory usage
- Configurable LoRA parameters (rank, alpha)
- Selective layer fine-tuning (vision/language layers)
- WandB integration for experiment tracking
- Automatic evaluation during training

**Main Functions:**
- `load_data()` - Load and prepare OCR dataset with conversation format
- `load_model()` - Load base model and configure LoRA
- `tune()` - Execute fine-tuning with specified hyperparameters

**Usage:**
```bash
# Basic training
python core/sft_gemma.py \
  --config configs/hyperparm_configs/default.json \
  --lora-output-dir models/gemma-ocr

# Custom hyperparameters
python core/sft_gemma.py \
  --learning-rate 2e-5 \
  --num-train-epochs 5 \
  --lora-r 32 \
  --lora-alpha 32
```

**Configuration Parameters:**
- `model_name` - Path to base model (default: `/tmp2/share_data/google--gemma-3-12b-it`)
- `input_data` - Path to training data JSON file
- `instruction` - OCR task instruction (default: `請給我 OCR 結果`)
- `learning_rate` - Learning rate for optimizer
- `num_train_epochs` - Number of training epochs
- `lora_r` - LoRA rank (lower = fewer parameters)
- `lora_alpha` - LoRA scaling parameter
- `max_seq_length` - Maximum sequence length (default: 8192)
- `load_in_4bit` - Enable 4-bit quantization
- `finetune_vision_layers` - Fine-tune vision tower (default: False)
- `finetune_language_layers` - Fine-tune language layers (default: True)

---

## Hyperparameter Search

### hyperparameter_search.py

Automated hyperparameter optimization using Optuna framework with multiple search strategies.

**Key Features:**
- **Multiple Search Strategies:**
  - `bayesian` - Bayesian Optimization with TPE sampler (most efficient, recommended)
  - `grid` - Exhaustive grid search over discrete values
  - `random` - Random sampling for quick exploration

- **Intelligent Pruning:**
  - Median pruner stops unpromising trials early
  - Configurable startup trials (5) and warmup steps (10)

- **Persistent Storage:**
  - SQLite database for study persistence
  - Resume capability for interrupted searches
  - Automatic saving of best configurations

- **Resource Management:**
  - Automatic GPU memory cleanup between trials
  - Error handling for failed trials (returns inf)
  - Per-trial model checkpoints and results

**Default Search Space:**
- `learning_rate`: loguniform(1e-5, 5e-4) - Log-uniform distribution for learning rates
- `num_train_epochs`: int(2, 5) - Integer range for number of epochs

**Command-line Arguments:**
```bash
--study-name          # Name of Optuna study (default: auto-generated timestamp)
--search-mode         # bayesian|random|grid (default: bayesian)
--n-trials           # Number of trials to run (default: 20)
--timeout            # Timeout in seconds (optional)
--resume             # Resume previous study with same name
--config             # JSON config with search_space and fixed params
--storage            # Optuna storage URL (default: SQLite)
--model-name         # Base model path
--input-data         # Training data path
--instruction        # OCR instruction text
--base-output-dir    # Base directory for trial models
```

**Config File Format:**
```json
{
  "search_space": {
    "learning_rate": {
      "type": "loguniform",
      "low": 1e-5,
      "high": 5e-4
    },
    "num_train_epochs": {
      "type": "int",
      "low": 2,
      "high": 5,
      "step": 1
    },
    "lora_r": {
      "type": "categorical",
      "choices": [8, 16, 32, 64]
    },
    "warmup_ratio": {
      "type": "uniform",
      "low": 0.0,
      "high": 0.15
    }
  },
  "train_split": "train[:90%]",
  "val_split": "train[90%:]",
  "max_seq_length": 8192,
  "load_in_4bit": true,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 4
}
```

**Supported Parameter Types:**
- `loguniform` - Log-uniform distribution (good for learning rates, spans orders of magnitude)
- `uniform` - Uniform distribution (for continuous values in linear scale)
- `int` - Integer range with optional step size
- `categorical` - Discrete choices from a list

**Usage Examples:**
```bash
# Quick start - Bayesian optimization with 20 trials
python core/hyperparameter_search.py \
  --study-name gemma-ocr-opt \
  --n-trials 20

# Grid search over specific ranges
python core/hyperparameter_search.py \
  --search-mode grid \
  --config configs/hyperparm_configs/grid_search_small.json

# Continue previous study
python core/hyperparameter_search.py \
  --study-name gemma-ocr-opt \
  --resume

# Random search with custom config
python core/hyperparameter_search.py \
  --study-name my-search \
  --search-mode random \
  --n-trials 30 \
  --config configs/hyperparm_configs/optuna/config.json
```

**Output Files:**
- `optuna_studies/{study_name}.db` - SQLite database with all trial data
- `optuna_studies/{study_name}_best_params.json` - Best parameters with metadata
- `optuna_studies/{study_name}_best_config.json` - Ready-to-use config file
- `models/hyperparam_search/trial_{n}/` - Per-trial model checkpoints
- `models/hyperparam_search/trial_{n}/trial_{n}_results.json` - Trial results

**Training with Best Parameters:**
```bash
# Use automatically generated best config
python core/sft_gemma.py \
  --config optuna_studies/gemma-ocr-opt_best_config.json \
  --lora-output-dir models/gemma-ocr-best
```

### analyze_hyperparam_results.py

Comprehensive analysis and visualization of hyperparameter search results.

**Key Features:**
- Parameter importance analysis (identifies which parameters matter most)
- Optimization history visualization (eval loss over trials)
- Parallel coordinate plots (visualize high-dimensional parameter relationships)
- Best trial comparison (top N trials with detailed metrics)
- Statistical summaries and reports
- Support for comparing multiple studies

**Usage:**
```bash
# Analyze a specific study
python core/analyze_hyperparam_results.py \
  --study-name gemma-ocr-opt

# Generate detailed report with visualizations
python core/analyze_hyperparam_results.py \
  --study-name gemma-ocr-opt \
  --output-dir reports/gemma-ocr-opt

# Compare multiple studies
python core/analyze_hyperparam_results.py \
  --study-name study1 \
  --compare study2 study3

# Launch interactive Optuna Dashboard
optuna-dashboard sqlite:///optuna_studies/gemma-ocr-opt.db
# Open browser at http://127.0.0.1:8080
```

**Output Visualizations:**
- Parameter importance plot
- Optimization history (eval_loss vs trial number)
- Parallel coordinate plot (all parameters vs eval_loss)
- Top trials comparison table

---

## Evaluation Scripts

### evaluate.py

Main evaluation script for OCR models using VLLM inference service.

**Key Features:**
- Parallel evaluation with configurable worker threads
- VLLM integration for fast inference
- Automatic image encoding (base64 data URLs)
- Progress saving with configurable intervals
- Support for resuming interrupted evaluations
- CER and F1 score calculation
- Detailed per-sample metrics

**Usage:**
```bash
# Basic evaluation
python core/evaluate.py

# Custom VLLM service and workers
python core/evaluate.py \
  --vllm-url http://192.168.1.78:3472/v1/chat/completions \
  --data-file data/input/ocr_test.json \
  --output-file results/gemma_evaluation.json \
  --workers 4 \
  --save-interval 50

# Evaluate all samples (no limit)
python core/evaluate.py --top-k None --workers 8

# Evaluate on benchmark dataset
python core/evaluate.py \
  --data-file data/benchmark/table.json \
  --output-file results/gemma_table.json \
  --model-name gemma-12b
```

**Parameters:**
- `--vllm-url` - VLLM service endpoint URL
- `--data-file` - Input JSON file with image paths and ground truth
- `--output-file` - Output JSON file for results
- `--top-k` - Number of samples to evaluate (None for all)
- `--workers` - Number of parallel worker threads
- `--model-name` - Model name for VLLM service
- `--save-interval` - Save progress every N samples

**Output Format:**
```json
{
  "overall_cer": 0.0234,
  "overall_f1": 0.9876,
  "total_samples": 100,
  "timestamp": "2025-12-22T10:30:00",
  "model_name": "gemma-12b",
  "samples": [
    {
      "image_path": "path/to/image.png",
      "label": "ground truth text",
      "prediction": "predicted text",
      "cer_score": 0.0123,
      "f1_score": 0.9890
    }
  ]
}
```

### count_score.py

Recalculate CER and F1 scores with text normalization.

**Key Features:**
- Text normalization (removes whitespace and punctuation)
- Handles both ASCII and Chinese punctuation
- Batch processing for efficiency
- Individual and overall metrics
- Preserves original predictions for comparison

**Normalization Process:**
1. Remove all whitespace (spaces, tabs, newlines)
2. Remove ASCII punctuation
3. Remove Chinese punctuation marks (，。！？etc.)

**Usage:**
```bash
# Recalculate with normalized text
python core/count_score.py \
  --input results/gemma_evaluation.json \
  --output results/gemma_evaluation_norm.json
```

**Why Normalize?**
- Focus on character recognition accuracy
- Reduce penalty for whitespace differences
- More robust comparison across different OCR systems
- Better reflects actual content accuracy

---

## Model Management

### gemma_adapter_merge.py

Merge LoRA adapters into base model for deployment with VLLM.

**Purpose:**
VLLM doesn't support separate adapter files for vision towers, so we need to merge the LoRA weights into the base model before deployment.

**Key Features:**
- Disk space checking (requires ~30GB)
- Direct adapter loading with FastLanguageModel
- 16-bit merged output (not 4-bit)
- Automatic chat template configuration
- Device auto-mapping for memory efficiency

**Usage:**
```bash
# Basic merge with defaults
python core/gemma_adapter_merge.py

# Custom paths
python core/gemma_adapter_merge.py \
  --lora_dir models/gemma-12b-sft-input_size=500 \
  --output_dir models/merged--gemma-3-12b-sft
```

**Parameters:**
- `--lora_dir` - Path to LoRA adapter directory (default: `models/gemma-12b-sft-input_size=500`)
- `--output_dir` - Output directory for merged model (default: `models/merged--gemma-3-12b-sft_input_size_500_test`)

**Process:**
1. Check available disk space (min 30GB)
2. Load LoRA adapter using FastLanguageModel
3. Configure chat template (gemma-3)
4. Merge and save as 16-bit model
5. Output ready for VLLM deployment

**Important Notes:**
- Uses 16-bit precision for best quality
- Requires significant disk space
- Device auto-mapping handles large models
- Output is directly compatible with VLLM

---

## Data Processing

### split_dataset.py

Split raw OCR dataset into training and test sets with reproducible random shuffling.

**Key Features:**
- Reproducible random splitting (seed=42)
- Configurable split ratio (default: 1:5 non_test:test)
- Data format conversion (img_path → image_path)
- Automatic output directory creation
- Detailed split statistics

**Split Ratio:**
- non_test: 1/6 of total data (for training/validation)
- test: 5/6 of total data (for evaluation)

**Usage:**
```bash
python core/split_dataset.py
```

**Input:** `data/raw_data/ocr.json`
```json
[
  {
    "img_path": "path/to/image.png",
    "ocr_results": "text content"
  }
]
```

**Output Files:**
- `data/input/ocr_non_test.json` - Training/validation data
- `data/input/ocr_test.json` - Test data

**Output Format:**
```json
[
  {
    "image_path": "path/to/image.png",
    "ocr_text": "text content"
  }
]
```

---

## Analysis Tools

### data_analysis/

Directory containing analysis and visualization tools for OCR results.

**Files:**
- `analyze_empty_predictions.py` - Analyze samples with empty predictions
- `train_test_data.py` - Training and test data statistics
- `output/` - Generated analysis outputs

**Usage:**
```bash
# Analyze empty predictions
python core/data_analysis/analyze_empty_predictions.py

# Generate data statistics
python core/data_analysis/train_test_data.py
```

---

## Best Practices

### For Training:
1. Start with hyperparameter search to find optimal parameters
2. Use 4-bit quantization for faster training on consumer GPUs
3. Monitor validation loss to prevent overfitting
4. Use WandB for experiment tracking and comparison

### For Hyperparameter Search:
1. Start with Bayesian optimization (most efficient)
2. Use 20-30 trials for good coverage
3. Define reasonable search spaces (not too wide)
4. Resume interrupted searches with `--resume`
5. Analyze results before final training

### For Evaluation:
1. Use multiple workers for faster evaluation
2. Save progress regularly with `--save-interval`
3. Normalize scores for fair comparison
4. Keep raw predictions for debugging

### For Deployment:
1. Merge LoRA adapters before VLLM deployment
2. Check disk space before merging
3. Use 16-bit merged models for best quality
4. Test merged model before full deployment

---

## Troubleshooting

### Out of Memory (OOM):
- Enable 4-bit quantization: `--load-in-4bit`
- Reduce batch size: `--per-device-train-batch-size 1`
- Increase gradient accumulation: `--gradient-accumulation-steps 8`
- Reduce LoRA rank: `--lora-r 8`

### Slow Training:
- Increase batch size if memory allows
- Use gradient accumulation for effective larger batches
- Check GPU utilization with `nvidia-smi`
- Reduce max_seq_length if not needed

### Hyperparameter Search Fails:
- Check storage path permissions
- Verify config JSON syntax
- Ensure enough disk space for trial checkpoints
- Use `--resume` to continue from last successful trial

### Evaluation Issues:
- Verify VLLM service is running: `curl {vllm_url}`
- Check image paths are accessible
- Reduce workers if getting timeouts
- Verify model name matches VLLM deployment

---

## File Dependencies

```
core/
├── sft_gemma.py                    # Required by: hyperparameter_search.py
├── hyperparameter_search.py        # Uses: sft_gemma.py
├── analyze_hyperparam_results.py   # Uses: optuna_studies/*.db
├── evaluate.py                     # Uses: count_score.py (for metrics)
├── count_score.py                  # Uses: lib/utils/count_metric.py
├── gemma_adapter_merge.py          # Standalone
├── split_dataset.py                # Standalone
└── data_analysis/                  # Analysis utilities
```

## Related Configuration Files

```
configs/hyperparm_configs/
├── default.json                    # Default training config
├── optuna/
│   └── config.json                # Hyperparameter search config
├── search_space_aggressive.json    # Wide search space
├── search_space_conservative.json  # Narrow search space
└── grid_search_small.json          # Small grid search
```
