"""
分析評估結果中 F1 score <= 0.6 的失敗樣本

用法:
    python core/data_analysis/failcase_analyze_ocrtext.py <result_json_path> [--output <output_path>] [--test-data <test_data_size>]

範例:
    python core/data_analysis/failcase_analyze_ocrtext.py results/evaluation/test_data=2400/gemma3_sft_lr2e4_ep5_test2700.json
    python core/data_analysis/failcase_analyze_ocrtext.py results/evaluation/test_data=2400/gemma3_sft_lr2e4_ep5_test2700.json --test-data 2400
    python core/data_analysis/failcase_analyze_ocrtext.py results/evaluation/test_data=2400/gemma3_sft_lr2e4_ep5_test2700.json --output custom_output.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


def analyze_failcases(result_path: str, output_path: str = None, threshold: float = 0.6) -> Dict[str, Any]:
    """
    分析評估結果中 F1 score <= threshold 的失敗樣本

    Args:
        result_path: 評估結果 JSON 檔案路徑
        output_path: 輸出檔案路徑，若為 None 則不寫入檔案
        threshold: F1 score 閾值（預設 0.6）

    Returns:
        包含分析結果的字典
    """
    # 讀取 JSON 檔案
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 取得 samples
    samples = data.get('samples', [])

    # 找出 F1 score <= threshold 的項目
    failcases = []
    for item in samples:
        if 'f1_score' in item:
            f1 = item['f1_score']
            # 檢查 F1 score 是否 <= threshold
            if f1 is not None and f1 <= threshold:
                failcases.append(item)

    # 統計數量
    count = len(failcases)
    total_count = len(samples)

    # 計算 F1 score 的分佈
    f1_ranges = {
        '0.0': 0,           # F1 == 0.0
        '0.0-0.2': 0,       # 0.0 < F1 <= 0.2
        '0.2-0.4': 0,       # 0.2 < F1 <= 0.4
        '0.4-0.6': 0,       # 0.4 < F1 <= 0.6
        '0.6-0.8': 0,       # 0.6 < F1 <= 0.8
        '0.8-1.0': 0,       # 0.8 < F1 <= 1.0
    }

    # 計算整體資料中各 focus_layout 的分佈（原分佈）
    all_layout_counts = {}
    for item in samples:
        focus_layouts = item.get('focus_layout', ['unknown'])
        if not isinstance(focus_layouts, list):
            focus_layouts = [focus_layouts]

        for layout in focus_layouts:
            if layout not in all_layout_counts:
                all_layout_counts[layout] = 0
            all_layout_counts[layout] += 1

    # 根據 focus_layout 分組
    layout_groups = {}

    # 統計所有樣本的 F1 分布（不只是失敗案例）
    for item in samples:
        f1 = item.get('f1_score', 0)
        if f1 == 0.0:
            f1_ranges['0.0'] += 1
        elif f1 <= 0.2:
            f1_ranges['0.0-0.2'] += 1
        elif f1 <= 0.4:
            f1_ranges['0.2-0.4'] += 1
        elif f1 <= 0.6:
            f1_ranges['0.4-0.6'] += 1
        elif f1 <= 0.8:
            f1_ranges['0.6-0.8'] += 1
        else:  # f1 > 0.8
            f1_ranges['0.8-1.0'] += 1

    # 分組失敗案例
    for item in failcases:

        # 分組 - focus_layout 是一個 list，需要遍歷
        focus_layouts = item.get('focus_layout', ['unknown'])
        if not isinstance(focus_layouts, list):
            focus_layouts = [focus_layouts]

        for layout in focus_layouts:
            if layout not in layout_groups:
                layout_groups[layout] = []
            layout_groups[layout].append(item)

    # 計算每個 focus_layout 的統計資料
    layout_stats = {}
    for focus_layout, items in layout_groups.items():
        total_layout_count = all_layout_counts.get(focus_layout, 0)
        layout_stats[focus_layout] = {
            'failcase_count': len(items),
            'failcase_percentage': round(len(items) / total_count * 100, 2) if total_count > 0 else 0,
            'total_count': total_layout_count,
            'total_percentage': round(total_layout_count / total_count * 100, 2) if total_count > 0 else 0,
            'fail_rate': round(len(items) / total_layout_count * 100, 2) if total_layout_count > 0 else 0,
            'avg_f1': round(sum(item.get('f1_score', 0) for item in items) / len(items), 4) if items else 0,
            'avg_cer': round(sum(item.get('cer', 0) for item in items) / len(items), 4) if items else 0
        }

    # 建立分析結果
    analysis_result = {
        'source_file': result_path,
        'total_samples': total_count,
        'failcase_count': count,
        'failcase_percentage': round(count / total_count * 100, 2) if total_count > 0 else 0,
        'pass_count': total_count - count,
        'threshold': threshold,
        'f1_distribution': f1_ranges,
        'focus_layout_distribution': layout_stats,
        'focus_layout_groups': {k: v for k, v in layout_groups.items()},
        'overall_cer': data.get('overall_cer'),
        'overall_f1': data.get('overall_f1'),
        'model_name': data.get('model_name'),
        'timestamp': data.get('timestamp'),
        'samples': failcases
    }

    # 輸出統計資訊
    print(f"{'='*60}")
    print(f"檔案: {result_path}")
    print(f"{'='*60}")
    print(f"總樣本數:           {total_count}")
    print(f"失敗樣本數 (F1 <= {threshold}): {count} ({analysis_result['failcase_percentage']}%)")
    print(f"通過樣本數 (F1 > {threshold}):  {total_count - count}")
    print(f"模型名稱:           {data.get('model_name', 'N/A')}")
    print(f"Overall CER:        {data.get('overall_cer', 'N/A')}")
    print(f"Overall F1:         {data.get('overall_f1', 'N/A')}")
    print(f"{'='*60}")

    # 顯示 F1 分佈（基於所有樣本）
    print(f"\nF1 Score 分佈 (所有樣本):")
    print(f"  F1 = 0.0:       {f1_ranges['0.0']:4d} ({round(f1_ranges['0.0']/total_count*100, 1) if total_count > 0 else 0}%)")
    print(f"  0.0 < F1 ≤ 0.2: {f1_ranges['0.0-0.2']:4d} ({round(f1_ranges['0.0-0.2']/total_count*100, 1) if total_count > 0 else 0}%)")
    print(f"  0.2 < F1 ≤ 0.4: {f1_ranges['0.2-0.4']:4d} ({round(f1_ranges['0.2-0.4']/total_count*100, 1) if total_count > 0 else 0}%)")
    print(f"  0.4 < F1 ≤ 0.6: {f1_ranges['0.4-0.6']:4d} ({round(f1_ranges['0.4-0.6']/total_count*100, 1) if total_count > 0 else 0}%)")
    print(f"  0.6 < F1 ≤ 0.8: {f1_ranges['0.6-0.8']:4d} ({round(f1_ranges['0.6-0.8']/total_count*100, 1) if total_count > 0 else 0}%)")
    print(f"  0.8 < F1 ≤ 1.0: {f1_ranges['0.8-1.0']:4d} ({round(f1_ranges['0.8-1.0']/total_count*100, 1) if total_count > 0 else 0}%)")

    # 顯示 Focus Layout 分佈對比
    print(f"\nFocus Layout 分佈對比:")
    print(f"  {'Layout':<15s} | {'原分佈':<20s} | {'Fail Case 分佈 (F1<0.6)':<20s} | {'失敗率':<10s} | {'Avg F1':<8s} | {'Avg CER':<8s}")
    print(f"  {'-'*15:15s} | {'-'*20:20s} | {'-'*20:20s} | {'-'*10:10s} | {'-'*8:8s} | {'-'*8:8s}")

    sorted_layouts = sorted(layout_stats.items(), key=lambda x: x[1]['failcase_count'], reverse=True)
    for focus_layout, stats in sorted_layouts:
        orig_dist = f"{stats['total_count']:4d} ({stats['total_percentage']:5.1f}%)"
        fail_dist = f"{stats['failcase_count']:4d} ({stats['failcase_percentage']:5.1f}%)"
        fail_rate = f"{stats['fail_rate']:5.1f}%"
        avg_f1 = f"{stats['avg_f1']:.4f}"
        avg_cer = f"{stats['avg_cer']:.4f}"
        print(f"  {focus_layout:<15s} | {orig_dist:<20s} | {fail_dist:<20s} | {fail_rate:<10s} | {avg_f1:<8s} | {avg_cer:<8s}")

    # 顯示前幾個失敗案例的範例
    if count > 0:
        print(f"\n前 5 個失敗案例:")
        for i, item in enumerate(failcases[:5]):
            img_path = item.get('image_path', 'N/A')
            f1 = item.get('f1_score', 'N/A')
            cer = item.get('cer', 'N/A')
            label_len = len(item.get('label', ''))
            pred_len = len(item.get('prediction', ''))

            # 格式化 F1 和 CER
            f1_str = f"{f1:.3f}" if isinstance(f1, (int, float)) else str(f1)
            cer_str = f"{cer:.3f}" if isinstance(cer, (int, float)) else str(cer)

            print(f"  [{i+1}] F1: {f1_str}, CER: {cer_str}")
            print(f"      圖片: {img_path}")
            print(f"      Label 長度: {label_len}, Prediction 長度: {pred_len}")

    # 寫入檔案
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)
        print(f"\n結果已寫入: {output_path}")

    return analysis_result


def main():
    parser = argparse.ArgumentParser(
        description='分析評估結果中 F1 score <= 0.6 的失敗樣本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--result_path',
        type=str,
        required=True,
        help='評估結果 JSON 檔案路徑'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='輸出檔案路徑（預設：不寫入檔案）'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.6,
        help='F1 score 閾值（預設：0.6）'
    )
    parser.add_argument(
        '--test-data',
        type=str,
        default=None,
        help='測試資料大小（例如：2400），用於指定輸出目錄。若未指定，會從輸入路徑自動提取'
    )

    args = parser.parse_args()

    # 檢查輸入檔案是否存在
    if not Path(args.result_path).exists():
        print(f"錯誤: 找不到檔案 {args.result_path}")
        return

    # 設定預設輸出路徑
    if args.output is None:
        # 從路徑中提取 test_data 大小，或使用 --test-data 參數
        test_data_size = args.test_data
        if test_data_size is None:
            # 嘗試從路徑中提取 test_data=xxxx
            result_path_obj = Path(args.result_path)
            for parent in result_path_obj.parents:
                if parent.name.startswith('test_data='):
                    test_data_size = parent.name.split('=')[1]
                    break

        # 如果仍然無法確定 test_data_size，使用預設值
        if test_data_size is None:
            test_data_size = "2400"

        # 建立輸出目錄
        output_dir = Path(f"results/evaluation/test_data={test_data_size}/failcase")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 使用輸入檔案名稱作為輸出檔案名稱
        input_stem = Path(args.result_path).stem
        args.output = str(output_dir / f"failcase_f1_{args.threshold}_{input_stem}.json")

    # 執行分析
    analyze_failcases(args.result_path, args.output, args.threshold)


if __name__ == '__main__':
    main()
