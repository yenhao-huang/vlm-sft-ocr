import yaml
import base64
import json
import requests
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset


def yml_parse(config_path: str) -> str:
    """
    解析 YAML 配置文件並返回 prompt_text

    Args:
        config_path: YAML 配置文件路徑

    Returns:
        prompt_text: OCR 提示文字
    """
    if config_path is None:
        return "執行 OCR 任務："

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        prompt_text = config.get('prompt_text', '執行 OCR 任務：')
        print(f"Loaded prompt from config: {config_path}")
        return prompt_text
    except FileNotFoundError:
        print(f"警告: 找不到配置文件 {config_path}，使用預設 prompt")
        return "執行 OCR 任務："
    except Exception as e:
        print(f"警告: 解析配置文件時發生錯誤 {e}，使用預設 prompt")
        return "執行 OCR 任務："

def _encode_image(image_path: str) -> str:
    """
    將圖片編碼為 base64 data URL

    Args:
        image_path: 圖片檔案路徑

    Returns:
        base64 編碼的 data URL
    """
    # 如果已經是 URL（http/https/data/file），直接返回
    if image_path.startswith(('http://', 'https://', 'data:', 'file://')):
        return image_path

    # 讀取圖片並轉為 base64
    with open(image_path, 'rb') as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')

    # 根據檔案副檔名決定 MIME 類型
    ext = Path(image_path).suffix.lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp'
    }
    mime_type = mime_types.get(ext, 'image/png')

    return f"data:{mime_type};base64,{image_data}"

def call_vllm_stream(vllm_url: str, image_url: str, model_name: str, prompt_text: str = "執行 OCR 任務：") -> str:
    """
    使用 requests 串流方式呼叫 VLLM 對單張圖片執行 OCR

    Args:
        vllm_url: VLLM 服務的 URL (例如: http://192.168.1.78:3472/v1/chat/completions)
        image_url: 圖片的 URL 或路徑
        model_name: 模型名稱
        prompt_text: OCR 提示文字 (預設: "執行 OCR 任務：")

    Returns:
        生成的 OCR 文字
    """
    encoded_image = _encode_image(image_url)

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
                        "text": prompt_text
                    }
                ]
            }
        ],
        "temperature": 0.7,
#        "top_p": 0.9,
#        "frequency_penalty": 0.1,
#        "presence_penalty": 0.1,
        "repetition_penalty": 1.1,
        "stop": ["<end_of_turn>"],
    }

    headers = {
        "Authorization": "Bearer EMPTY",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            vllm_url,
            headers=headers,
            json=payload,
            stream=True,
            timeout=60
        )
        response.raise_for_status()

        # 處理串流響應
        full_text = ""
        chunk_count = 0
        content_count = 0

        for line in response.iter_lines():
            if line:  # 過濾掉空行
                chunk_count += 1
                decoded_line = line.decode("utf-8")
                if decoded_line.startswith("data:"):
                    data = decoded_line[5:].strip()  # 移除 "data:" 前綴
                    if data == "[DONE]":  # 串流結束
                        break
                    try:
                        # 解析 JSON 數據
                        chunk = json.loads(data)
                        content = chunk["choices"][0]["delta"].get("content", "")
                        if content:
                            content_count += 1
                            # 實時輸出串流內容
                            print(content, end="", flush=True)
                            full_text += content
                    except json.JSONDecodeError:
                        continue

        # 串流結束後換行
        if content_count > 0:
            print()  # 換行

        # 調試信息
        if chunk_count == 0:
            print(f"警告：沒有收到任何串流數據")
        elif content_count == 0:
            print(f"警告：收到 {chunk_count} 個 chunk 但沒有內容")

        return full_text

    except requests.exceptions.Timeout:
        print(f"請求超時，返回空字串")
        return ""
    except requests.exceptions.HTTPError as e:
        print(f"HTTP 錯誤：{e}")
        print(f"請求 URL：{vllm_url}")
        return ""
    except Exception as e:
        print(f"未預期的錯誤：{e}")
        return ""
    

def single_ocr(vllm_url: str, image_path: str, model_name: str, mode: str = "vllm", prompt_text: str = "執行 OCR 任務：") -> str:
    """
    對單張圖片執行 OCR

    Args:
        vllm_url: API 服務的 URL
        image_path: 圖片路徑
        model_name: 模型名稱
        mode: 呼叫模式，可選值:
            - "vllm": 使用 call_vllm (預設)
            - "vllm_stream": 使用 call_vllm_stream
            - "local_api": 使用 call_local_api
        prompt_text: OCR 提示文字 (預設: "執行 OCR 任務：")

    Returns:
        OCR 結果文字
    """
    if mode == "vllm_stream":
        return call_vllm_stream(vllm_url, image_path, model_name, prompt_text)
    else:
        raise ValueError(f"不支援的模式: {mode}，請使用 'vllm', 'vllm_stream', 或 'local_api'")

def batch_ocr(test_dataset, vllm_url: str = "http://192.168.1.78:3132/v1/chat/completions", model_name: str = "mistralai3", workers: int = 1, save_interval: int = 0, checkpoint_file: str = None, mode: str = "vllm", prompt_text: str = "執行 OCR 任務："):
    """
    批次執行 OCR

    Args:
        test_dataset: 測試資料集
        vllm_url: API 服務的 URL
        model_name: 模型名稱
        workers: 並行處理的 worker 數量
        save_interval: 每處理多少個樣本就保存一次中間結果（0 表示不保存中間結果）
        checkpoint_file: 中間結果保存路徑（如果為 None 則自動生成）
        mode: 呼叫模式 ("vllm", "vllm_stream", "local_api")
        prompt_text: OCR 提示文字 (預設: "執行 OCR 任務：")

    Returns:
        包含所有結果的列表
    """
    def process_sample(sample):
        """處理單個樣本"""
        prediction = single_ocr(vllm_url, sample["image_path"], model_name, mode=mode, prompt_text=prompt_text)

    all_results = []
    for idx, sample in enumerate(tqdm(test_dataset, desc="Processing OCR"), 1):
        result = process_sample(sample)
        all_results.append(result)

    return all_results

def load_dataset_fn(data_file="data/input/ocr_test.json", top_k=None):
    """
    Load test dataset for evaluation.
    """
    test_dataset = load_dataset(
        "json",
        data_files={"test": data_file},
    )["test"]

    eval_dataset = test_dataset.select(range(min(top_k, len(test_dataset)))) if top_k else test_dataset
    print(f"Evaluating on {len(eval_dataset)} samples...")

    return eval_dataset

def load_images_from_dir(image_dir: str):
    """
    從指定目錄載入所有圖片

    Args:
        image_dir: 圖片目錄路徑

    Returns:
        圖片路徑列表
    """
    image_dir_path = Path(image_dir)
    if not image_dir_path.exists():
        raise ValueError(f"目錄不存在: {image_dir}")

    # 支援的圖片格式
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}

    # 收集所有圖片檔案
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(image_dir_path.glob(f'*{ext}'))
        image_paths.extend(image_dir_path.glob(f'*{ext.upper()}'))

    # 排序以確保順序一致
    image_paths = sorted(image_paths)

    print(f"找到 {len(image_paths)} 張圖片在 {image_dir}")
    return [str(p) for p in image_paths]

def batch_ocr_from_dir(
    image_dir: str,
    vllm_url: str = "http://192.168.1.78:3132/v1/chat/completions",
    model_name: str = "mistralai3",
    mode: str = "vllm_stream",
    prompt_text: str = "執行 OCR 任務：",
    config_path: str = None
):
    """
    從圖片目錄批次執行 OCR 並印出結果

    Args:
        image_dir: 圖片目錄路徑
        vllm_url: API 服務的 URL
        model_name: 模型名稱
        mode: 呼叫模式 (預設: "vllm_stream")
        prompt_text: OCR 提示文字
        config_path: YAML 配置文件路徑（可選）

    Returns:
        包含所有結果的列表，每個元素是 dict: {"image_path": str, "ocr_result": str}
    """
    # 如果提供了 config_path，從配置文件讀取 prompt
    if config_path:
        prompt_text = yml_parse(config_path)

    # 載入圖片
    image_paths = load_images_from_dir(image_dir)

    if not image_paths:
        print("沒有找到任何圖片")
        return []

    # 批次處理
    results = []
    for image_path in tqdm(image_paths, desc="OCR Processing"):
        print(f"\n{'='*80}")
        print(f"處理圖片: {Path(image_path).name}")
        print(f"{'='*80}")

        ocr_result = single_ocr(
            vllm_url=vllm_url,
            image_path=image_path,
            model_name=model_name,
            mode=mode,
            prompt_text=prompt_text
        )

        result_entry = {
            "image_path": image_path,
            "image_name": Path(image_path).name,
            "ocr_result": ocr_result
        }
        results.append(result_entry)
        '''
        print(f"\n{'-'*80}")
        print(f"結果: {ocr_result}")
        print(f"{'-'*80}\n")
        '''
    # 印出摘要
    print(f"\n{'='*80}")
    print(f"完成！共處理 {len(results)} 張圖片")
    print(f"{'='*80}\n")

    return results

def run(
    vllm_url="http://192.168.1.78:3132/v1/chat/completions",
    data_file="data/input/ocr_test.json",
    top_k=None,
    model_name="mistralai3",
    output_file="results/mistral_evaluation_results.json",
    workers=1,
    save_interval=0,
    checkpoint_file=None,
    mode="vllm",
    prompt_text="執行 OCR 任務："
):
    """
    Run OCR evaluation on the test dataset.

    Args:
        vllm_url: API 服務的 URL
        data_file: 測試資料檔案路徑
        top_k: 要評估的樣本數量（None 表示全部）
        model_name: 模型名稱
        output_file: 輸出檔案路徑
        workers: 並行處理的 worker 數量
        save_interval: 每處理多少個樣本就保存一次中間結果（0 表示不保存中間結果）
        checkpoint_file: 中間結果保存路徑
        mode: 呼叫模式 ("vllm", "vllm_stream", "local_api")
        prompt_text: OCR 提示文字 (預設: "執行 OCR 任務：")
    """
    # Load dataset
    print("Loading dataset...")
    test_dataset = load_dataset_fn(data_file=data_file, top_k=top_k)

    # Run batch OCR
    print(f"Running batch OCR with {workers} worker(s) using mode: {mode}...")
    print(f"Using prompt: {prompt_text}")
    if save_interval > 0:
        print(f"中間結果將每 {save_interval} 個樣本保存一次")
    results = batch_ocr(
        test_dataset,
        vllm_url=vllm_url,
        model_name=model_name,
        workers=workers,
        save_interval=save_interval,
        checkpoint_file=checkpoint_file,
        mode=mode,
        prompt_text=prompt_text
    )

    return results


if __name__ == "__main__":
    import argparse

    # 解析命令列參數
    parser = argparse.ArgumentParser(description='批次 OCR 處理')
    parser.add_argument('--image_dir', type=str, default='data/benchmark/mistral_benchmark',
                        help='圖片目錄路徑')
    parser.add_argument('--vllm_url', type=str, default='http://192.168.1.78:3132/v1/chat/completions',
                        help='VLLM API URL')
    parser.add_argument('--model_name', type=str, default='mistralai3',
                        help='模型名稱')
    parser.add_argument('--mode', type=str, default='vllm_stream',
                        choices=['vllm_stream'],
                        help='執行模式')
    parser.add_argument('--prompt_config', type=str, default='configs/ocr_prompt/add_lang_constraint.yml',
                        help='Prompt 配置文件路徑')
    parser.add_argument('--output_file', type=str, default='results/batch_ocr_results.json',
                        help='輸出結果檔案路徑')
    args = parser.parse_args()

    print(f"圖片目錄: {args.image_dir}")
    print(f"VLLM URL: {args.vllm_url}")
    print(f"Model: {args.model_name}")
    print(f"Mode: {args.mode}")
    print(f"Prompt Config: {args.prompt_config}\n")

    results = batch_ocr_from_dir(
        image_dir=args.image_dir,
        vllm_url=args.vllm_url,
        model_name=args.model_name,
        mode=args.mode,
        config_path=args.prompt_config
    )

    # 保存結果
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n結果已保存到: {args.output_file}")