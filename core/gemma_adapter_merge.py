"""
合併 Gemma LoRA adapter 到 base model

修正版本：使用 FastLanguageModel.from_pretrained 直接載入 adapter
"""

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
import argparse
import shutil
import sys


def check_disk_space(path="/tmp2", required_gb=30):
    """檢查指定路徑是否有足夠的磁碟空間"""
    stat = shutil.disk_usage(path)
    available_gb = stat.free / (1024 ** 3)

    print(f"檢查磁碟空間: {path}")
    print(f"  可用空間: {available_gb:.2f} GB")
    print(f"  需要空間: {required_gb} GB")

    if available_gb < required_gb:
        print(f"\n❌ 錯誤: {path} 只有 {available_gb:.2f} GB 可用空間，需要至少 {required_gb} GB")
        sys.exit(1)

    print(f"✓ 磁碟空間充足\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="合併 Gemma LoRA adapter 到 base model")
    parser.add_argument(
        "--lora_dir",
        type=str,
        default="models/gemma-12b-sft-input_size=500",
        help="LoRA adapter 目錄路徑"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/merged--gemma-3-12b-sft_input_size_500_test",
        help="合併後模型的輸出目錄"
    )
    args = parser.parse_args()

    # 檢查磁碟空間
    check_disk_space()

    max_seq_length = 8192
    dtype = None
    load_in_4bit = False  # 合併時不要用 4bit，使用 16bit

    print("載入模型和 LoRA adapter...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.lora_dir,  # 直接指向 adapter 目錄
        max_seq_length=max_seq_length,
        dtype=dtype,
        device_map="auto",
        load_in_4bit=load_in_4bit,
        attn_implementation="eager",
    )

    print("設定 chat template...")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma-3",
    )

    print(f"開始合併並儲存到: {args.output_dir}")
    model.save_pretrained_merged(
        args.output_dir,
        tokenizer,
        save_method="merged_16bit",
    )

    print("✓ 合併完成！")


if __name__ == "__main__":
    main()
