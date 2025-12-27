import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import json
import base64

def _encode_image(image_path: str) -> str:
    """將圖片編碼為 base64 data URL"""
    if image_path.startswith(('http://', 'https://', 'data:', 'file://')):
        return image_path

    with open(image_path, 'rb') as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')

    ext = Path(image_path).suffix.lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
    }
    mime_type = mime_types.get(ext, 'image/png')

    return f"data:{mime_type};base64,{image_data}"

def test_stream(vllm_url: str, image_path: str, model_name: str):
    """測試串流響應"""
    encoded_image = _encode_image(image_path)

    payload = {
        "model": model_name,
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": encoded_image}
                    },
                    {
                        "type": "text",
                        "text": "執行 OCR 任務："
                    }
                ]
            }
        ],
        "temperature": 0.7,
        "repetition_penalty": 1.1,
        "stop": ["<end_of_turn>"],
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    print(f"發送請求到: {vllm_url}")
    print(f"模型: {model_name}")
    print(f"串流模式: True")
    print("-" * 50)

    try:
        response = requests.post(
            vllm_url,
            json=payload,
            headers=headers,
            stream=True,
            timeout=60
        )
        response.raise_for_status()

        print(f"響應狀態碼: {response.status_code}")
        print(f"響應 headers: {dict(response.headers)}")
        print("-" * 50)
        print("開始接收串流數據:")
        print("-" * 50)

        full_text = ""
        line_count = 0

        for line in response.iter_lines():
            line_count += 1
            if line:
                line_str = line.decode('utf-8')
                print(f"[行 {line_count}] 原始: {line_str[:100]}...")

                if line_str.startswith('data: '):
                    data_str = line_str[6:]
                    if data_str == '[DONE]':
                        print("收到 [DONE] 信號")
                        break
                    try:
                        data = json.loads(data_str)
                        print(f"[行 {line_count}] 解析 JSON: {json.dumps(data, ensure_ascii=False)[:200]}...")

                        if 'choices' in data and len(data['choices']) > 0:
                            delta = data['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                print(f"[行 {line_count}] 內容片段: '{content}'")
                                full_text += content
                    except json.JSONDecodeError as e:
                        print(f"[行 {line_count}] JSON 解析錯誤: {e}")
                        continue

        print("-" * 50)
        print(f"總共接收 {line_count} 行")
        print(f"完整文字長度: {len(full_text)}")
        print(f"完整文字: {full_text}")
        return full_text

    except Exception as e:
        print(f"錯誤: {e}")
        import traceback
        traceback.print_exc()
        return ""

if __name__ == "__main__":
    # 測試參數
    vllm_url = "http://192.168.1.78:3472/v1/chat/completions"
    model_name = "gemma-l2b"

    # 需要一個測試圖片路徑
    image_path = "data/pdf2png/mohw/(檔案下載)衛生福利部111年10月20日公告/(檔案下載)衛生福利部111年10月20日公告_page_0001.png"

    if Path(image_path).exists():
        result = test_stream(vllm_url, image_path, model_name)
    else:
        print(f"圖片不存在: {image_path}")
        print("請指定一個有效的圖片路徑")
