import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import base64
import requests
import json
import argparse
import yaml
import time
from datetime import datetime
from datasets import load_dataset
from core.count_score import count_score
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

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

def call_local_api(url: str, image_url: str, model_name: str = None) -> str:
    """
    呼叫本地 API 對單張圖片執行 OCR

    Args:
        url: API 服務的 URL (例如: http://192.168.1.76:3669/predict)
        image_url: 圖片的 URL 或路徑
        model_name: 模型名稱 (optional, 此 API 可能不需要)

    Returns:
        生成的 OCR 文字
    """
    # 確保 URL 包含 http/https 協議
    if not url.startswith(('http://', 'https://')):
        url = f"http://{url}"

    try:
        encoded_image = _encode_image(image_url)
        img_bytes = base64.b64decode(encoded_image.split(",")[1])
        files = {"file": ("image.jpg", img_bytes, "image/jpeg")}

        # 發送 POST 請求
        response = requests.post(
            url,
            files=files,
            timeout=60
        )
        response.raise_for_status()

        # 解析回應並返回 text 欄位
        result = response.json()
        return result.get("text", "")

    except requests.exceptions.Timeout:
        print(f"請求超時，返回空字串")
        return ""
    except requests.exceptions.HTTPError as e:
        print(f"HTTP 錯誤：{e}")
        print(f"請求 URL：{url}")
        print(f"回應內容：{response.text if 'response' in locals() else 'N/A'}")
        return ""
    except KeyError as e:
        print(f"回應中缺少 'text' 欄位：{e}")
        print(f"回應內容：{result if 'result' in locals() else 'N/A'}")
        return ""
    except Exception as e:
        print(f"未預期的錯誤：{e}")
        return ""

def call_vllm(vllm_url: str, image_url: str, model_name: str, prompt_text: str = "執行 OCR 任務：") -> str:
    """
    呼叫 VLLM 方法對單張圖片執行 OCR

    Args:
        vllm_url: VLLM 服務的 URL
        image_url: 圖片的 URL 或路徑
        model_name: 模型名稱
        prompt_text: OCR 提示文字 (預設: "執行 OCR 任務：")

    Returns:
        生成的 OCR 文字
    """
    encoded_image = _encode_image(image_url)

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": encoded_image
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            }
        ],
        "temperature": 0.7,
        "repetition_penalty": 1.1,
        "stop": ["<end_of_turn>"],
    }


    try:
        response = requests.post(
            vllm_url,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except requests.exceptions.Timeout:
        print(f"請求超時，返回空字串")
        return ""
    except requests.exceptions.HTTPError as e:
        print(f"HTTP 錯誤：{e}")
        print(f"請求 URL：{vllm_url}")
        print(f"回應內容：{response.text}")
        return ""
    except Exception as e:
        print(f"未預期的錯誤：{e}")
        return ""

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
    if mode == "vllm":
        return call_vllm(vllm_url, image_path, model_name, prompt_text)
    elif mode == "vllm_stream":
        return call_vllm_stream(vllm_url, image_path, model_name, prompt_text)
    elif mode == "local_api":
        return call_local_api(vllm_url, image_path, model_name)
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
        包含所有結果的列表和時間統計資訊的元組 (results, time_stats)
    """
    def process_sample(sample):
        """處理單個樣本"""
        start_time = time.time()
        prediction = single_ocr(vllm_url, sample["image_path"], model_name, mode=mode, prompt_text=prompt_text)
        end_time = time.time()
        elapsed_time = end_time - start_time

        ground_truth = sample["ocr_text"]
        image_path = sample["image_path"]

        return {
            "prediction": prediction,
            "ground_truth": ground_truth,
            "image_path": image_path,
            "inference_time": elapsed_time
        }

    def save_checkpoint(results, checkpoint_path):
        """保存中間檢查點"""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "processed_samples": len(results),
            "results": results
        }
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        print(f"\n已保存中間結果到 {checkpoint_path}（已處理 {len(results)} 個樣本）")

    all_results = []

    # 設定檢查點檔案路徑
    if save_interval > 0 and checkpoint_file is None:
        checkpoint_file = f"results/eval_checkpoint/checkpoint_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    if workers == 1:
        # 單線程處理
        for idx, sample in enumerate(tqdm(test_dataset, desc="Processing OCR"), 1):
            result = process_sample(sample)
            all_results.append(result)

            # 定期保存中間結果
            if save_interval > 0 and idx % save_interval == 0:
                save_checkpoint(all_results, checkpoint_file)
    else:
        # 多線程並行處理
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_sample, sample): sample for sample in test_dataset}

            for idx, future in enumerate(tqdm(as_completed(futures), total=len(test_dataset), desc="Processing OCR"), 1):
                try:
                    result = future.result()
                    all_results.append(result)

                    # 定期保存中間結果
                    if save_interval > 0 and idx % save_interval == 0:
                        save_checkpoint(all_results, checkpoint_file)
                except Exception as e:
                    sample = futures[future]
                    print(f"\nError processing {sample['image_path']}: {e}")

    # 計算時間統計
    inference_times = [res["inference_time"] for res in all_results]
    time_stats = {
        "average_time_per_sample": sum(inference_times) / len(inference_times) if inference_times else 0,
        "total_time": sum(inference_times),
        "min_time": min(inference_times) if inference_times else 0,
        "max_time": max(inference_times) if inference_times else 0
    }

    return all_results, time_stats

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


def save(results, scores, model_name, output_file="results/evaluation_results.json", time_stats=None):
    """
    Save evaluation results to a JSON file.

    Args:
        results: List of dictionaries containing prediction, ground_truth, and image_path
        scores: Dictionary containing overall and individual scores
        model_name: Name of the model used
        output_file: Path to save the results
        time_stats: Dictionary containing timing statistics (optional)
    """
    predictions = [res["prediction"] for res in results]
    references = [res["ground_truth"] for res in results]
    image_paths = [res["image_path"] for res in results]

    output = {
        "overall_cer": scores["overall_cer"],
        "overall_f1": scores["overall_f1"],
        "total_samples": len(predictions),
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "average_inference_time": time_stats["average_time_per_sample"] if time_stats else None,
        "total_inference_time": time_stats["total_time"] if time_stats else None,
        "min_inference_time": time_stats["min_time"] if time_stats else None,
        "max_inference_time": time_stats["max_time"] if time_stats else None,
        "samples": [
            {
                "image_path": img_path,
                "label": label,
                "prediction": pred,
                "cer_score": individual_cer,
                "f1_score": individual_f1,
                "inference_time": res.get("inference_time")
            }
            for img_path, label, pred, individual_cer, individual_f1, res in zip(
                image_paths,
                references,
                predictions,
                scores["individual_cer_scores"],
                scores["individual_f1_scores"],
                results
            )
        ]
    }

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save to JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {output_file}")
    print(f"Saved {len(output['samples'])} samples with predictions, labels, image paths, CER, and F1 scores")

    return output_file

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
    results, time_stats = batch_ocr(
        test_dataset,
        vllm_url=vllm_url,
        model_name=model_name,
        workers=workers,
        save_interval=save_interval,
        checkpoint_file=checkpoint_file,
        mode=mode,
        prompt_text=prompt_text
    )

    # Calculate scores
    print("Calculating scores...")
    scores = count_score(results)

    print(f"\n{'='*50}")
    print(f"Character Error Rate (CER): {scores['overall_cer']:.4f} ({scores['overall_cer']*100:.2f}%)")
    print(f"F1 Score: {scores['overall_f1']:.4f} ({scores['overall_f1']*100:.2f}%)")
    print(f"Total samples evaluated: {len(results)}")
    print(f"Average inference time per sample: {time_stats['average_time_per_sample']:.4f} seconds")
    print(f"Total inference time: {time_stats['total_time']:.2f} seconds")
    print(f"{'='*50}")

    # Save results
    save(results=results, scores=scores, model_name=model_name, output_file=output_file, time_stats=time_stats)

    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OCR evaluation using VLLM service")
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://192.168.1.76:3132/v1/chat/completions",
        help="VLLM service URL (default: http://192.168.1.78:3132/v1/chat/completions)"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/benchmark/ocr_test.json",
        help="Path to test data file (default: data/input/ocr_test.json)"
    )
    parser.add_argument(
        "--top-k",
        type=lambda x: None if x.lower() == 'none' else int(x),
        default=None,
        help="Number of samples to evaluate (default: 100, use 'None' to evaluate all)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="mistralai3",
        help="Model name (default: mistralai3)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results/mistral_evaluation_results.json",
        help="Path to output results file (default: results/mistral_evaluation_results.json)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel workers for processing (default: 1)"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=0,
        help="Save checkpoint every N samples (0 = disable checkpointing, default: 0)"
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default=None,
        help="Path to save checkpoint file (default: auto-generate in results/ folder)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="vllm",
        choices=["vllm", "vllm_stream", "local_api"],
        help="API call mode: 'vllm' (default), 'vllm_stream', or 'local_api'"
    )
    parser.add_argument(
        "--prompt-config",
        type=str,
        default="configs/ocr_prompt/default.yml",
        help="Path to YAML config file for prompt (e.g., configs/ocr_prompt/default.yml). If not specified, uses default prompt."
    )

    args = parser.parse_args()

    # Parse prompt from config file
    prompt_text = yml_parse(args.prompt_config)

    run(
        vllm_url=args.vllm_url,
        data_file=args.data_file,
        top_k=args.top_k,
        model_name=args.model_name,
        output_file=args.output_file,
        workers=args.workers,
        save_interval=args.save_interval,
        checkpoint_file=args.checkpoint_file,
        mode=args.mode,
        prompt_text=prompt_text
    )