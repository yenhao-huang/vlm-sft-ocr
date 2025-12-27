import torch
import argparse
import json
import os
from datetime import datetime
from unsloth import FastLanguageModel, FastVisionModel
from unsloth.chat_templates import get_chat_template
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import TextStreamer


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
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        attn_implementation="eager",
    )

    print("Configuring LoRA for vision fine-tuning...")

    model = FastLanguageModel.get_peft_model(
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
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

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

    del trainer

    return trainer_stats, final_eval_results


def inference(
    model,
    tokenizer,
    image_path: str,
    instruction: str = "請給我 OCR 結果",
    max_new_tokens: int = 128,
    temperature: float = 1.5,
    min_p: float = 0.1,
    stream: bool = True
):
    """
    Run inference on a single image.

    Args:
        model: The model to use for inference
        tokenizer: The tokenizer
        image_path: Path to the image file
        instruction: Instruction text for OCR
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        min_p: Minimum probability for sampling
        stream: Whether to stream the output

    Returns:
        Generated text or None if streaming
    """
    print(f"Running inference on: {image_path}")
    FastVisionModel.for_inference(model)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]
        }
    ]

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image_path,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    if stream:
        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=temperature,
            min_p=min_p
        )
        return None
    else:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=temperature,
            min_p=min_p
        )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Vision OCR Fine-tuning with Unsloth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/hyperparm_configs/default.json",
        help="Path to config JSON file"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="/tmp2/share_data/google--gemma-3-12b-it",
        help="Path or name of the base model"
    )
    parser.add_argument(
        "--input_data",
        type=str,
        default="data/input/ocr_non_test.json",
        help="Path to input JSON data file"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="請給我 OCR 結果",
        help="Instruction text for OCR task"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/test_input_size=500",
        help="Output directory for training artifacts"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size per device for training"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=8192,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=True,
        help="Load model in 4-bit quantization"
    )
    parser.add_argument(
        "--close_report_to_wandb",
        action="store_true",
        help="Report training metrics to Weights & Biases"
    )
    parser.add_argument(
        "--is_debug",
        action="store_true",
        help="Enable debug mode"
    )
    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)

        # Get default values from parser
        defaults = {}
        for action in parser._actions:
            if action.dest != 'help' and action.dest != 'config':
                defaults[action.dest] = action.default

        # Only override with config if the arg wasn't explicitly provided on command line
        for key, value in config.items():
            # If the current value equals the default, use config value
            # Otherwise, keep the command line value
            if hasattr(args, key) and getattr(args, key) == defaults.get(key):
                setattr(args, key, value)

    return args


if __name__ == "__main__":
    args = parse_args()

    print("="*80)
    print("Configuration:")
    print("="*80)
    print(f"  Model: {args.model_name}")
    print(f"  Input Data: {args.input_data}")
    print(f"  Instruction: {args.instruction}")
    print(f"  Output Dir: {args.output_dir}")
    if args.config:
        print(f"  Config File: {args.config}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Epochs: {args.num_train_epochs}")
    print(f"  Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
    print(f"  Batch Size: {args.per_device_train_batch_size}")
    print(f"  LoRA r: {args.lora_r}")
    print(f"  LoRA alpha: {args.lora_alpha}")
    print(f"  Max Seq Length: {args.max_seq_length}")
    print("="*80)

    # Create output directories
    lora_output_dir = os.path.join(args.output_dir, "lora")
    os.makedirs(lora_output_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    raw_train_dataset, raw_val_dataset, train_dataset, val_dataset = load_data(
        input_data=args.input_data,
        instruction=args.instruction
    )

    # Load model
    model, tokenizer = load_model(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )

    # Test inference before training
    if args.is_debug:
        print("\n" + "="*80)
        print("Testing inference BEFORE training:")
        print("="*80)
        inference(model, tokenizer, raw_train_dataset[0]["image_path"], args.instruction)

    # Fine-tune
    print("\n" + "="*80)
    print("Starting fine-tuning:")
    print("="*80)
    trainer_stats, final_eval_results = tune(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        report_to_wandb=args.close_report_to_wandb == False,
        warmup_ratio=0.0,
        max_seq_length=args.max_seq_length
    )

    # Test inference after training
    if args.is_debug:
        print("\n" + "="*80)
        print("Testing inference AFTER training:")
        print("="*80)
        inference(model, tokenizer, raw_train_dataset[0]["image_path"], args.instruction)

    # Save model (LoRA adapters)
    print("\n" + "="*80)
    print(f"Saving LoRA model to: {lora_output_dir}")
    print("="*80)
    model.save_pretrained(lora_output_dir)
    tokenizer.save_pretrained(lora_output_dir)
    print("LoRA model saved successfully!")
