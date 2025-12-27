import os
# 禁用 torch.compile 以避免編譯快取問題
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from unsloth import FastLanguageModel, FastVisionModel
from datasets import load_dataset

# Load dataset for test
raw_train_dataset, raw_val_dataset = load_dataset(
    "json",
    data_files="/tmp2/howard/vl-sft-ocr/data/input/example.json",
    split=["train[:90%]", "train[90%:]"],
)

# Model configuration
max_seq_length = 8192
dtype = None
load_in_4bit = True

#model_name = "/tmp2/howard/vl-sft-ocr/models/merged--gemma-3-12b-sft_input_size_500"
model_name = "/tmp2/howard/vl-sft-ocr/models/merged--gemma-3-hyperparam-search"
print(f"Loading merged model from: {model_name}")
print("="*80)

try:
    # 載入模型 - 使用 fp16 而非量化以避免問題
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,
        device_map="auto",
        attn_implementation="eager",
    )

    print("✅ Model loaded successfully!")
    print(f"Model type: {type(model).__name__}")
    print("="*80)

    # Enable inference mode
    FastVisionModel.for_inference(model)
    print("✅ Inference mode enabled!")
    print("="*80)

    # Prepare test input
    image = raw_train_dataset[0]["image_path"]
    instruction = "請給我 OCR 結果"

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]}
    ]

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    print(f"Test image: {image}")
    print(f"Instruction: {instruction}")
    print("="*80)
    print("Generating output...")
    print("="*80)

    # Generate output with streaming
    from transformers import TextStreamer
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=128,
        use_cache=True,
        temperature=1.5,
        min_p=0.1
    )

    print("="*80)
    print("✅ Merge validation PASSED! Model works correctly.")

except Exception as e:
    print(f"❌ Merge validation FAILED!")
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
