"""
使用純 Transformers 進行 Gemma 模型推理

不依賴 unsloth，使用標準的 Transformers API
"""

import os

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from accelerate import Accelerator
from PIL import Image
import argparse


def main():
    parser = argparse.ArgumentParser(description="Gemma OCR Inference with Transformers")
    parser.add_argument(
        "--model_name",
        type=str,
        default="models/merged--gemma-3-12b-sft_lr1e5_ep5",
        help="合併後的模型路徑"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="./data/pdf2png/moea/1111207162322696_1/1111207162322696_1_page_0001.png",
        help="圖片路徑"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="請給我 OCR 結果",
        help="指令文字"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="最大生成 token 數"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="生成溫度"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="模型精度"
    )
    args = parser.parse_args()

    # 設定 dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # 載入模型
    print("="*60)
    print("載入模型...")
    print("="*60)
    print(f"模型: {args.model_name}")
    print(f"精度: {args.dtype}")

    torch_device = Accelerator().device

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=dtype
    )

    print(f"✓ 模型已載入到 {model.device}")

    # 準備輸入
    print("\n" + "="*60)
    print("準備輸入...")
    print("="*60)
    print(f"圖片: {args.image_path}")
    print(f"指令: {args.instruction}")

    # 檢查圖片是否存在
    if not os.path.exists(args.image_path):
        print(f"✗ 錯誤: 找不到圖片 {args.image_path}")
        return

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": args.image_path},
                {"type": "text", "text": args.instruction}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        padding=True,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=dtype)

    # 生成
    print("\n" + "="*60)
    print("開始推理...")
    print("="*60)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            temperature=args.temperature,
            min_p=0.1,
            do_sample=True if args.temperature > 0 else False,
        )

    # 解碼結果
    result = processor.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    # 輸出結果
    print(f"\n{'='*60}")
    print("OCR 結果:")
    print(f"{'-'*60}")
    print(result)
    print(f"{'-'*60}\n")


if __name__ == "__main__":
    main()
