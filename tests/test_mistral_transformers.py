import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from accelerate import Accelerator
import argparse


def main():
    parser = argparse.ArgumentParser(description="OCR Inference")
    parser.add_argument("--model_name", type=str, default="/tmp2/share_data/unsloth--Ministral-3-14B-Instruct-2512-FP8/")
    parser.add_argument("--image_path", type=str, default="./data/pdf2png/moea/1111207162322696_1/1111207162322696_1_page_0001.png")
    parser.add_argument("--instruction", type=str, default="請給我 OCR 結果")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    # Load model
    print("Loading model...")
    torch_device = Accelerator().device
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForImageTextToText.from_pretrained(args.model_name, device_map=torch_device, dtype=auto)


    # Prepare input
    messages = [{"role": "user", "content": [
        {"type": "image", "url": args.image_path},
        {"type": "text", "text": args.instruction}
    ]}]
    inputs = processor.apply_chat_template(messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

    # Generate
    print(f"Running inference on: {args.image_path}")
    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, use_cache=True, temperature=args.temperature, min_p=0.1)
    result = processor.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Output
    print(f"\n{'='*60}\nOCR Result:\n{'-'*60}\n{result}\n{'-'*60}\n")


if __name__ == "__main__":
    main()
