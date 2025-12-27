#!/usr/bin/env python3
"""
Hyperparameter search for OCR fine-tuning using Optuna.

This script enables automatic hyperparameter optimization using different search strategies:
- Bayesian Optimization (default, recommended)
- Grid Search (exhaustive but slower)
- Random Search (faster but less efficient)

Example usage:
    # Bayesian optimization with 20 trials
    python core/hyperparameter_search.py --n-trials 20 --study-name gemma-ocr-opt

    # Grid search over specific ranges
    python core/hyperparameter_search.py --search-mode grid --config configs/grid_search.json

    # Continue previous study
    python core/hyperparameter_search.py --study-name gemma-ocr-opt --resume
"""

import argparse
import json
import os
import sys
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler, GridSampler, RandomSampler
import multiprocessing as mp
import torch
torch._dynamo.disable()

# Import training functions
from unsloth import FastLanguageModel, FastVisionModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

def cleanup_memory():
    """Thoroughly clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        # 2. 同步 GPU 任務，確保之前的計算已結束
        torch.cuda.synchronize() 
        # 3. 釋放 PyTorch 佔用的快取顯存
        torch.cuda.empty_cache()
        # 4. 再次清理與重置統計數據
        torch.cuda.reset_peak_memory_stats()
        gc.collect()

    print_memory_stats()

def print_memory_stats():
    if torch.cuda.is_available():
        # Print current memory usage
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(i) / 1024**3    # GB
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")

def load_data(input_data: str, instruction: str, train_split: str = "train[:90%]", val_split: str = "train[90%:]"):
    """
    Load and prepare dataset for vision OCR fine-tuning.

    Args:
        input_data: Path to JSON data file
        instruction: Instruction text for OCR task
        train_split: Training data split ratio
        val_split: Validation data split ratio

    Returns:
        tuple: (raw_train_dataset, raw_val_dataset, train_dataset, val_dataset)
    """
    print(f"Loading dataset from: {input_data}")
    raw_train_dataset, raw_val_dataset = load_dataset(
        "json",
        data_files=input_data,
        split=[train_split, val_split],
    )

    def convert_to_conversation(sample):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": sample["image_path"]}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample["ocr_text"]}
                ]
            },
        ]
        return {"messages": conversation}

    # Filter out samples with empty ocr_text
    filtered_train = [sample for sample in raw_train_dataset if sample["ocr_text"] and sample["ocr_text"].strip()]
    filtered_val = [sample for sample in raw_val_dataset if sample["ocr_text"] and sample["ocr_text"].strip()]

    train_dataset = [convert_to_conversation(sample) for sample in filtered_train]
    val_dataset = [convert_to_conversation(sample) for sample in filtered_val]

    print(f"Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val samples")
    print(f"Filtered out: {len(raw_train_dataset) - len(train_dataset)} train, {len(raw_val_dataset) - len(val_dataset)} val samples with empty ocr_text")
    return raw_train_dataset, raw_val_dataset, train_dataset, val_dataset


def load_model(
    model_name: str,
    max_seq_length: int = 8192,
    load_in_4bit: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 16,
    finetune_vision_layers: bool = False,
    finetune_language_layers: bool = True,
    chat_template: str = "gemma-3"
):
    """
    Load and configure model for vision fine-tuning.

    Args:
        model_name: Path or name of the base model
        max_seq_length: Maximum sequence length
        load_in_4bit: Whether to load model in 4-bit quantization
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        finetune_vision_layers: Whether to fine-tune vision layers
        finetune_language_layers: Whether to fine-tune language layers
        chat_template: Chat template name

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        attn_implementation="eager",
    )

    print("Configuring LoRA for vision fine-tuning...")

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=finetune_vision_layers,
        finetune_language_layers=finetune_language_layers,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=3407,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=chat_template,
    )

    print("Model loaded and configured successfully")
    return model, tokenizer



def tune(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    output_dir: str = "models/vision_sft",
    num_train_epochs: int = 5,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.1,
    eval_steps: int = 100,
    report_to_wandb: bool = True,
    save_steps: int = 100,
    max_seq_length: int = 8192
):
    """
    Fine-tune the model on OCR dataset.

    Args:
        model: The model to train
        tokenizer: The tokenizer
        train_dataset: Training dataset
        val_dataset: Validation dataset
        output_dir: Directory to save model checkpoints
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device for training
        per_device_eval_batch_size: Batch size per device for evaluation
        gradient_accumulation_steps: Number of gradient accumulation steps
        learning_rate: Learning rate
        warmup_ratio: Warmup ratio for learning rate scheduler
        eval_steps: Evaluation frequency
        report_to_wandb: Whether to report metrics to Weights & Biases
        save_steps: Save checkpoint frequency
        max_seq_length: Maximum sequence length

    Returns:
        tuple: (trainer_stats, final_eval_results)
            - trainer_stats: Training statistics from trainer.train()
            - final_eval_results: Final evaluation metrics from trainer.evaluate()
    """
    print("Preparing trainer...")
    FastVisionModel.for_training(model)

    if report_to_wandb:
        import wandb
        wandb.init(
            project="vlm-ocr-sft",
            name=f"gemma12b_lr{learning_rate}_epoch{num_train_epochs}",
            group="bbox-representation",
            notes="""
            bbox normalized to [0,1000]
            vision SFT with UnslothVisionDataCollator
            eval on MOHW dataset
            """,
        )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            metric_for_best_model="eval_loss",
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            logging_steps=1,
            logging_dir=f"{output_dir}/logs",
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=output_dir,
            report_to="wandb" if report_to_wandb else None,
            # Required for vision finetuning
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=max_seq_length,
        ),
    )

    # Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print_memory_stats()

    print("Starting training...")
    trainer_stats = trainer.train()

    # Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    # Save training metadata to JSON
    metadata = {
        "training_completed_at": datetime.now().isoformat(),
        "gpu_info": {
            "name": gpu_stats.name,
            "total_memory_gb": max_memory
        },
        "training_time": {
            "total_seconds": trainer_stats.metrics['train_runtime'],
            "total_minutes": round(trainer_stats.metrics['train_runtime'] / 60, 2),
            "total_hours": round(trainer_stats.metrics['train_runtime'] / 3600, 2)
        },
        "memory_stats": {
            "start_memory_gb": start_gpu_memory,
            "peak_memory_gb": used_memory,
            "peak_memory_for_training_gb": used_memory_for_lora,
            "peak_memory_percentage": used_percentage,
            "peak_training_memory_percentage": lora_percentage
        }
    }

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\nMetadata saved to: {metadata_path}")

    # Evaluate on validation set to get final metrics
    print("\n" + "="*80)
    print("Running final evaluation on validation set...")
    print("="*80)

    # Switch model to inference mode before evaluation
    FastVisionModel.for_inference(model)

    # Remove wandb callback before final evaluation to avoid logging errors
    if report_to_wandb:
        import wandb
        # Remove WandbCallback from trainer's callback list
        from transformers.integrations import WandbCallback
        trainer.callback_handler.callbacks = [
            cb for cb in trainer.callback_handler.callbacks
            if not isinstance(cb, WandbCallback)
        ]
        wandb.finish()

    final_eval_results = trainer.evaluate()

    print("Final Evaluation Results:")
    for key, value in final_eval_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    print("="*80)
    print(final_eval_results)

    del model
    del tokenizer
    del trainer
    return trainer_stats, final_eval_results



class HyperparameterSearch:
    """Hyperparameter search manager using Optuna."""

    def __init__(
        self,
        study_name: str,
        storage: Optional[str] = None,
        search_mode: str = "bayesian",
        config: Optional[Dict[str, Any]] = None,
        resume: bool = False
    ):
        """
        Initialize hyperparameter search.

        Args:
            study_name: Name of the Optuna study
            storage: Database URL for storing study results (default: SQLite)
            search_mode: Search strategy - 'bayesian', 'random', or 'grid'
            config: Configuration dict with 'search_space' and other fixed parameters
            resume: Whether to resume an existing study
        """
        self.study_name = study_name
        self.search_mode = search_mode

        # Parse config: separate search_space from base_config
        if config is None:
            config = {}

        self.param_config = config.get("search_space", self._default_param_config())
        self.base_config = {k: v for k, v in config.items() if k != "search_space"}

        # Setup storage
        if storage is None:
            os.makedirs("optuna_studies", exist_ok=True)
            storage = f"sqlite:///optuna_studies/{study_name}.db"
        self.storage = storage

        # Create or load study
        self.study = self._create_study(resume)

    def _default_param_config(self) -> Dict[str, Any]:
        """Default hyperparameter search space."""
        return {
            "learning_rate": {
                "type": "loguniform",
                "low": 1e-5,
                "high": 5e-4
            },
            "num_train_epochs": {
                "type": "int",
                "low": 2,
                "high": 5
            }
        }

    def _create_study(self, resume: bool) -> optuna.Study:
        """Create or load Optuna study."""
        # Select sampler based on search mode
        if self.search_mode == "bayesian":
            sampler = TPESampler(seed=3407)
        elif self.search_mode == "random":
            sampler = RandomSampler(seed=3407)
        elif self.search_mode == "grid":
            # For grid search, we need to specify all combinations
            search_space = self._create_grid_search_space()
            sampler = GridSampler(search_space)
        else:
            raise ValueError(f"Unknown search mode: {self.search_mode}")

        # Use median pruner to stop unpromising trials early
        pruner = MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=10
        )

        # Create or load study
        load_if_exists = resume
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction="minimize",  # Minimize eval_loss
            sampler=sampler,
            pruner=pruner,
            load_if_exists=load_if_exists
        )

        return study

    def _create_grid_search_space(self) -> Dict[str, list]:
        """Create search space for grid search."""
        search_space = {}
        for param_name, param_def in self.param_config.items():
            if param_def["type"] == "categorical":
                search_space[param_name] = param_def["choices"]
            elif param_def["type"] == "int":
                # For int, create a list of values
                low, high = param_def["low"], param_def["high"]
                step = param_def.get("step", 1)
                search_space[param_name] = list(range(low, high + 1, step))
            else:
                raise ValueError(f"Grid search doesn't support type: {param_def['type']}")
        return search_space

    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial."""
        params = {}

        for param_name, param_def in self.param_config.items():
            param_type = param_def["type"]

            if param_type == "loguniform":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_def["low"],
                    param_def["high"],
                    log=True
                )
            elif param_type == "uniform":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_def["low"],
                    param_def["high"]
                )
            elif param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_def["low"],
                    param_def["high"],
                    step=param_def.get("step", 1)
                )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_def["choices"]
                )
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")

        return params

    def train_worker(self, queue, config, trial_number):
        """
        在獨立進程中運行的訓練函數
        """
        try:
            # 1. Load data
            _, _, train_dataset, val_dataset = load_data(
                input_data=config.get("input_data", "data/input/ocr_non_test_data=500.json"),
                instruction=config.get("instruction", "請給我 OCR 結果"),
                train_split=config.get("train_split", "train[:90%]"),
                val_split=config.get("val_split", "train[90%:]")
            )

            # 2. Load model
            model, tokenizer = load_model(
                model_name=config.get("model_name", "/tmp2/share_data/google--gemma-3-12b-it"),
                max_seq_length=config.get("max_seq_length", 8192),
                load_in_4bit=config.get("load_in_4bit", True),
                lora_r=config.get("lora_r", 16),
                lora_alpha=config.get("lora_alpha", 16)
            )

            output_dir = f"{config.get('base_output_dir', 'models/hyperparam_search')}/trial_{trial_number}"
            os.makedirs(output_dir, exist_ok=True)

            # 3. Fine-tune
            trainer_stats, final_eval_results = tune(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                output_dir=output_dir,
                num_train_epochs=config["num_train_epochs"],
                per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
                gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
                learning_rate=config["learning_rate"],
                warmup_ratio=config.get("warmup_ratio", 0.1),
                max_seq_length=config.get("max_seq_length", 8192),
                eval_steps=config.get("eval_steps", 100),
                save_steps=config.get("save_steps", 100),
                report_to_wandb=config.get("report_to_wandb", True)
            )

            eval_loss = final_eval_results.get('eval_loss', float('inf'))
            
            # 清理並將結果放入隊列
            cleanup_memory()
            queue.put({"eval_loss": eval_loss, "success": True})

        except Exception as e:
            print(f"Error in worker process: {e}")
            import traceback
            traceback.print_exc()
            queue.put({"eval_loss": float('inf'), "success": False})

    def objective(self, trial: optuna.Trial) -> float:
        # 1. 建議參數
        params = self._suggest_hyperparameters(trial)
        config = {**self.base_config, **params}

        print("\n" + "="*80)
        print(f"Trial {trial.number}: Starting Spawned Process")
        print("="*80)

        # 2. 使用 Multiprocessing 隔離訓練
        # 使用 'spawn' 模式是 CUDA 必須的
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        
        p = ctx.Process(
            target=self.train_worker, 
            args=(queue, config, trial.number)
        )
        
        p.start()
        
        # 獲取結果
        try:
            result = queue.get(timeout=7200) # 設置一個合理的超時（如 2 小時）
            eval_loss = result["eval_loss"]
        except Exception as e:
            print(f"Trial {trial.number} failed to get result from queue: {e}")
            eval_loss = float('inf')
        finally:
            p.join() # 確保進程結束
            if p.is_alive():
                p.terminate()
        
        # 此時進程已關閉，1.5GB 的損耗會被系統自動釋放
        print(f"Trial {trial.number} finished. Process joined and memory released.")
        return eval_loss

    def optimize(self, n_trials: int = 20, timeout: Optional[int] = None):
        """
        Run hyperparameter optimization.

        Args:
            n_trials: Number of trials to run
            timeout: Timeout in seconds (optional)
        """
        print(f"\nStarting hyperparameter search: {self.study_name}")
        print(f"Search mode: {self.search_mode}")
        print(f"Number of trials: {n_trials}")
        print(f"Storage: {self.storage}\n")

        self.study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )

        print("\n" + "="*80)
        print("Optimization completed!")
        print("="*80)
        print(f"Best trial: {self.study.best_trial.number}")
        print(f"Best eval_loss: {self.study.best_value}")
        print("\nBest hyperparameters:")
        for key, value in self.study.best_params.items():
            print(f"  {key}: {value}")
        print("="*80)

        # Save best parameters
        self._save_best_params()

    def _save_best_params(self):
        """Save best parameters to file."""
        output_dir = "optuna_studies"
        os.makedirs(output_dir, exist_ok=True)

        best_params_file = f"{output_dir}/{self.study_name}_best_params.json"
        best_params = {
            "study_name": self.study_name,
            "best_trial": self.study.best_trial.number,
            "best_value": self.study.best_value,
            "best_params": self.study.best_params,
            "timestamp": datetime.now().isoformat()
        }

        with open(best_params_file, "w") as f:
            json.dump(best_params, f, indent=2)

        print(f"\nBest parameters saved to: {best_params_file}")

        # Also save as a config file ready to use
        config_file = f"{output_dir}/{self.study_name}_best_config.json"
        with open(config_file, "w") as f:
            json.dump(self.study.best_params, f, indent=2)
        print(f"Config file saved to: {config_file}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for OCR fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--study-name",
        type=str,
        default=f"ocr_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Name of the Optuna study"
    )
    parser.add_argument(
        "--search-mode",
        type=str,
        choices=["bayesian", "random", "grid"],
        default="bayesian",
        help="Search strategy"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of trials to run"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume previous study with same name"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hyperparm_configs/optuna/default.json",
        help="Path to JSON config file with 'search_space' and fixed parameters"
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (default: SQLite)"
    )

    # Base training parameters
    parser.add_argument(
        "--model-name",
        type=str,
        default="/tmp2/share_data/google--gemma-3-12b-it",
        help="Path or name of the base model"
    )
    parser.add_argument(
        "--input-data",
        type=str,
        default="data/input/ocr_non_test_data=500.json",
        help="Path to input JSON data file"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="請給我 OCR 結果",
        help="Instruction text for OCR task"
    )
    parser.add_argument(
        "--base-output-dir",
        type=str,
        default="models/hyperparam_search",
        help="Base directory for saving trial models"
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Set environment variables to prevent CUDA memory fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    args = parse_args()

    # Load config if provided
    config = {
        "model_name": args.model_name,
        "input_data": args.input_data,
        "instruction": args.instruction,
        "base_output_dir": args.base_output_dir
    }

    if args.config:
        with open(args.config, 'r') as f:
            config.update(json.load(f))

    # Create search manager
    search = HyperparameterSearch(
        study_name=args.study_name,
        storage=args.storage,
        search_mode=args.search_mode,
        config=config,
        resume=args.resume
    )

    # Run optimization
    search.optimize(
        n_trials=args.n_trials,
        timeout=args.timeout
    )
