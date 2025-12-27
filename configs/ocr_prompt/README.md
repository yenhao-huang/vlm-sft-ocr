# OCR Prompt Configuration

此目錄包含用於 OCR 評估的提示詞配置文件。

## 配置文件格式

每個配置文件都是一個 YAML 文件，包含以下欄位：

```yaml
# OCR Prompt Configuration
prompt_text: "您的提示詞文字"
```

## 管理 Prompt 配置

### 使用網頁界面（推薦）

啟動 Prompt Manager 網頁界面：

```bash
python ui/prompt_manager.py
```

這將在瀏覽器中打開一個管理界面，您可以：
- 📋 查看所有 prompt 配置
- ➕ 新增 prompt 配置
- ✏️ 更新現有配置
- 🗑️ 刪除配置

可選參數：
```bash
# 指定端口
python ui/prompt_manager.py --port 8080

# 創建公開分享鏈接
python ui/prompt_manager.py --share

# 指定配置目錄
python ui/prompt_manager.py --config-dir path/to/configs
```

### 手動創建配置文件

您也可以手動創建 YAML 配置文件，例如 `custom.yml`：

```yaml
# Custom OCR Prompt Configuration
prompt_text: "請詳細辨識圖片中的所有文字內容："
```

## 可用的配置文件範例

- `1.yml` - 預設中文提示詞：「執行 OCR 任務：」
- `2.yml` - 詳細中文提示詞：「請仔細辨識並輸出這張圖片中的所有文字內容：」
- `english.yml` - 英文提示詞：「Perform OCR task:」

## 在評估中使用 Prompt

在運行評估時，使用 `--prompt-config` 參數指定配置文件：

```bash
# 使用配置文件 1
python core/evaluate.py --prompt-config configs/ocr_prompt/1.yml --model-name mistralai3

# 使用配置文件 2
python core/evaluate.py --prompt-config configs/ocr_prompt/2.yml --model-name mistralai3

# 使用英文提示詞
python core/evaluate.py --prompt-config configs/ocr_prompt/english.yml --model-name mistralai3

# 不指定配置文件（使用預設提示詞）
python core/evaluate.py --model-name mistralai3
```
