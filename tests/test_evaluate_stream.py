import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.evaluate import call_vllm_stream

# 測試參數
vllm_url = "http://192.168.1.78:3472/v1/chat/completions"
model_name = "gemma-12b"
image_path = "data/pdf2png/mohw/(檔案下載)衛生福利部111年10月20日公告/(檔案下載)衛生福利部111年10月20日公告_page_0001.png"

print(f"測試 call_vllm_stream 函數")
print(f"URL: {vllm_url}")
print(f"模型: {model_name}")
print(f"圖片: {image_path}")
print("-" * 80)

result = call_vllm_stream(vllm_url, image_path, model_name)

print("-" * 80)
print(f"結果長度: {len(result)}")
print(f"結果內容:\n{result}")
