"""
分析評估結果中 prediction 為空的樣本

用法:
    python core/analyze_empty_predictions.py <result_json_path> [--output <output_path>]

範例:
    python core/analyze_empty_predictions.py results/gemma3_sft_lr2e4_ep5_test2700.json
    python core/analyze_empty_predictions.py results/gemma3_sft_lr2e4_ep5_test2700.json --output core/empty_analysis.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


def analyze_empty_predictions(result_path: str, output_path: str = None) -> Dict[str, Any]:
    """
    分析評估結果中 prediction 為空的樣本

    Args:
        result_path: 評估結果 JSON 檔案路徑
        output_path: 輸出檔案路徑，若為 None 則不寫入檔案

    Returns:
        包含分析結果的字典
    """
    # 讀取 JSON 檔案
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 取得 samples
    samples = data.get('samples', [])

    # 找出 prediction 為空的項目
    empty_predictions = []
    for idx, item in enumerate(samples):
        if 'prediction' in item:
            pred = item['prediction']
            # 檢查是否為空字串、None、或只有空白字元
            if pred is None or pred == '' or (isinstance(pred, str) and pred.strip() == ''):
                empty_predictions.append({
                    'index': idx,
                    'data': item
                })

    # 統計數量
    count = len(empty_predictions)
    total_count = len(samples)

    # 建立分析結果
    analysis_result = {
        'source_file': result_path,
        'total_samples': total_count,
        'empty_prediction_count': count,
        'empty_percentage': round(count / total_count * 100, 2) if total_count > 0 else 0,
        'non_empty_count': total_count - count,
        'overall_cer': data.get('overall_cer'),
        'overall_f1': data.get('overall_f1'),
        'model_name': data.get('model_name'),
        'timestamp': data.get('timestamp'),
        'empty_predictions': empty_predictions
    }

    # 輸出統計資訊
    print(f"{'='*60}")
    print(f"檔案: {result_path}")
    print(f"{'='*60}")
    print(f"總樣本數:           {total_count}")
    print(f"空 prediction 數量: {count} ({analysis_result['empty_percentage']}%)")
    print(f"非空 prediction:    {total_count - count}")
    print(f"模型名稱:           {data.get('model_name', 'N/A')}")
    print(f"Overall CER:        {data.get('overall_cer', 'N/A')}")
    print(f"Overall F1:         {data.get('overall_f1', 'N/A')}")
    print(f"{'='*60}")

    # 顯示前幾個空 prediction 的範例
    if count > 0:
        print(f"\n前 5 個空 prediction 的圖片路徑:")
        for i, item in enumerate(empty_predictions[:5]):
            img_path = item['data'].get('image_path', 'N/A')
            label_len = len(item['data'].get('label', ''))
            print(f"  [{i+1}] {img_path} (label 長度: {label_len})")

    # 寫入檔案
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)
        print(f"\n結果已寫入: {output_path}")

    return analysis_result


def main():
    parser = argparse.ArgumentParser(
        description='分析評估結果中 prediction 為空的樣本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'result_path',
        type=str,
        help='評估結果 JSON 檔案路徑'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='輸出檔案路徑（預設：不寫入檔案）'
    )

    args = parser.parse_args()

    # 檢查輸入檔案是否存在
    if not Path(args.result_path).exists():
        print(f"錯誤: 找不到檔案 {args.result_path}")
        return

    # 設定預設輸出路徑
    if args.output is None:
        input_stem = Path(args.result_path).stem
        args.output = f"core/empty_predictions_{input_stem}.json"

    # 執行分析
    analyze_empty_predictions(args.result_path, args.output)


if __name__ == '__main__':
    main()
