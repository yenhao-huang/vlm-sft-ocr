# Data Analysis Tools

此目錄包含用於分析模型評估結果的工具腳本。

## 腳本列表

### 1. analyze_empty_predictions.py
分析評估結果中 prediction 為空的樣本。

**用法：**
```bash
python core/data_analysis/analyze_empty_predictions.py <result_json_path> [--output <output_path>]
```

**範例：**
```bash
python core/data_analysis/analyze_empty_predictions.py results/gemma3_sft_lr2e4_ep5_test2700.json
python core/data_analysis/analyze_empty_predictions.py results/gemma3_sft_lr2e4_ep5_test2700.json --output core/empty_analysis.json
```

**輸出內容：**
- 總樣本數
- 空 prediction 數量與百分比
- 非空 prediction 數量
- 模型名稱與整體指標
- 前 5 個空 prediction 的圖片路徑

---

### 2. eval_result_analysis.py
分析評估結果中 F1 score <= 0.6 的失敗樣本。

requirement: analysis 前記得載入有 layout 的 results

**用法：**
```bash
python core/add_layout.py
python core/data_analysis/eval_result_analysis.py <result_json_path> [--output <output_path>] [--threshold <threshold>]
```

**參數：**
- `result_json_path`: 評估結果 JSON 檔案路徑
- `--output, -o`: 輸出檔案路徑（預設：`core/data_analysis/output/failcase_f1_{threshold}_{filename}.json`）
- `--threshold, -t`: F1 score 閾值（預設：0.6）

**範例：**
```bash
# 使用預設閾值 0.6
python core/data_analysis/eval_result_analysis.py results/gemma3_sft_lr2e4_ep5_test2700.json

# 自訂閾值為 0.5
python core/data_analysis/eval_result_analysis.py results/gemma3_sft_lr2e4_ep5_test2700.json --threshold 0.5

# 指定輸出路徑
python core/data_analysis/eval_result_analysis.py results/gemma3_sft_lr2e4_ep5_test2700.json --output custom_output.json
```

**輸出內容：**
- 總樣本數
- 失敗樣本數 (F1 <= threshold) 與百分比
- 通過樣本數 (F1 > threshold)
- F1 score 分佈統計（0.0, 0.0-0.2, 0.2-0.4, 0.4-0.6）
- 模型名稱與整體指標
- 前 5 個失敗案例的詳細資訊（F1, CER, 圖片路徑, label/prediction 長度）

---

### 3. train_test_data.py
處理訓練與測試資料集。

---

## 輸出目錄

分析結果預設會輸出到 `core/data_analysis/output/` 目錄。