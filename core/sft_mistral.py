"""
Fine-tuning script for Mistral 3 vision model for OCR task.
Converted from Jupyter notebook.
"""
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import torch
import numpy as np
from transformers import (
    Mistral3ForConditionalGeneration,
    FineGrainedFP8Config,
    AutoProcessor,
    AutoTokenizer,
)
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_model_and_processor(model_path: str, use_lora: bool = True):
    """Load model, processor and tokenizer.

    Args:
        model_path: Path to the model
        use_lora: If True, apply LoRA adapters to the model
    """
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load base model with quantization
    device_map = "auto"
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=FineGrainedFP8Config(dequantize=True),
    )

    # Apply LoRA if enabled
    if use_lora:
        # Prepare model for training with quantization
        #model = prepare_model_for_kbit_training(model)

        # Configure LoRA
        lora_config = LoraConfig(
            r=16,  # LoRA rank
            lora_alpha=32,  # LoRA scaling factor
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, processor, tokenizer


def load_and_prepare_datasets(data_path: str, instruction: str):
    """Load and prepare training and validation datasets."""
    raw_train_dataset, raw_val_dataset = load_dataset(
        "json",
        data_files=data_path,
        split=["train[:90%]", "train[90%:]"],
    )

    def convert_to_conversation(sample):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": sample["image_path"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["ocr_text"]}],
            },
        ]
        return {"messages": conversation}

    train_dataset = [convert_to_conversation(sample) for sample in raw_train_dataset]
    val_dataset = [convert_to_conversation(sample) for sample in raw_val_dataset]

    return train_dataset, val_dataset, raw_train_dataset


def test_inference(model, processor, tokenizer, image_path: str, instruction: str):
    """Test model inference on a single image."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": instruction},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    generate_ids = model.generate(**inputs, max_new_tokens=2000)
    decoded_output = tokenizer.decode(
        generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    return decoded_output


def preprocess_function(examples, processor, max_length: int = 4096):
    """Preprocess data, convert messages to model input."""
    messages = examples["messages"]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )

    # Labels are the same as input_ids for causal LM
    inputs["labels"] = inputs["input_ids"].clone()

    return inputs


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune Mistral 3 vision model for OCR task")

    parser.add_argument("--model_path", type=str, default="/tmp2/share_data/mistralai--Ministral-3-14B-Instruct-2512/")
    parser.add_argument("--data_path", type=str, default="data/input/ocr_non_test_data=500.json")
    parser.add_argument("--instruction", type=str, default="請給我 OCR 結果")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--output_dir", type=str, default="models/mistral_data=500")

    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Configuration
    model_path = args.model_path
    data_path = args.data_path
    instruction = args.instruction
    max_seq_length = args.max_seq_length
    output_dir = args.output_dir

    print("="*80)
    print("Configuration:")
    print("="*80)
    print(f"  Model Path: {args.model_path}")
    print(f"  Data Path: {args.data_path}")
    print(f"  Instruction: {args.instruction}")
    print(f"  Max Seq Length: {args.max_seq_length}")
    print(f"  Output Dir: {args.output_dir}")
    print("="*80)

    # Load and prepare datasets
    print("Loading datasets...")
    train_dataset, val_dataset, raw_train_dataset = load_and_prepare_datasets(
        data_path, instruction
    )

    # Load model and processor
    print("Loading model and processor...")
    model, processor, tokenizer = load_model_and_processor(
        model_path, use_lora=True
    )


    # Test inference (optional)
    '''
    print("\nTesting inference...")
    test_image = raw_train_dataset[0]["image_path"]
    output = test_inference(model, processor, tokenizer, test_image, instruction)
    print(f"Test output: {output[:200]}...")
    '''
    
    # Preprocess datasets
    print("\nPreprocessing datasets...")
    train_dataset = [
        preprocess_function(example, processor, max_seq_length)
        for example in train_dataset
    ]
    val_dataset = [
        preprocess_function(example, processor, max_seq_length)
        for example in val_dataset
    ]

    # Setup trainer
    print("\nSetting up trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        args=SFTConfig(
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=1000,
            metric_for_best_model="eval_loss",
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=output_dir,
            report_to="none",
            # Required for vision finetuning
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=max_seq_length,
        ),
    )

    # Train
    print("\nStarting training...")
    trainer_stats = trainer.train()

    print("\nTraining completed!")

    # Test inference (optional)
    print("\nTesting inference...")
    test_image = raw_train_dataset[0]["image_path"]
    output = test_inference(model, processor, tokenizer, test_image, instruction)
    print(f"Test output: {output[:200]}...")

    return trainer_stats


if __name__ == "__main__":
    main()
