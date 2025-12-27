#!/usr/bin/env python3
"""
Prompt Manager UI - Web Interface
管理 OCR Prompt 配置文件的網頁工具

功能：
- 列出所有 prompt 配置
- 新增 prompt 配置
- 更新 prompt 配置
- 刪除 prompt 配置
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import yaml
import gradio as gr
from datetime import datetime
from typing import List, Tuple


class PromptManager:
    """Prompt 配置管理器"""

    def __init__(self, config_dir: str = "configs/ocr_prompt"):
        """
        初始化 Prompt Manager

        Args:
            config_dir: 配置文件目錄路徑
        """
        self.config_dir = Path(config_dir)
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True, exist_ok=True)

    def get_prompt_list(self) -> List[str]:
        """
        獲取所有 prompt 配置文件名

        Returns:
            配置文件名列表
        """
        yaml_files = sorted(self.config_dir.glob("*.yml"))
        # 過濾掉 README
        return [f.name for f in yaml_files if f.name != "README.md"]

    def load_prompt(self, filename: str) -> Tuple[str, str]:
        """
        加載 prompt 配置

        Args:
            filename: 配置文件名

        Returns:
            (prompt_text, message)
        """
        if not filename:
            return "", "Please select a prompt config file"

        config_path = self.config_dir / filename

        if not config_path.exists():
            return "", f"❌ Config file '{filename}' does not exist"

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            prompt_text = config.get('prompt_text', '')
            return prompt_text, f"✅ Loaded: {filename}"

        except Exception as e:
            return "", f"❌ Error loading config: {str(e)}"

    def save_prompt(self, filename: str, prompt_text: str, is_new: bool = False) -> str:
        """
        保存 prompt 配置

        Args:
            filename: 配置文件名
            prompt_text: Prompt 文字
            is_new: 是否為新建文件

        Returns:
            結果訊息
        """
        if not filename:
            return "❌ Please enter a filename"

        if not prompt_text:
            return "❌ Please enter prompt text"

        # 確保副檔名為 .yml
        if not filename.endswith('.yml'):
            filename += '.yml'

        config_path = self.config_dir / filename

        # 檢查文件是否已存在（新建模式）
        if is_new and config_path.exists():
            return f"❌ Config file '{filename}' already exists. Use Update mode instead."

        # 檢查文件是否不存在（更新模式）
        if not is_new and not config_path.exists():
            return f"❌ Config file '{filename}' does not exist. Use Add mode instead."

        try:
            config = {'prompt_text': prompt_text}

            with open(config_path, 'w', encoding='utf-8') as f:
                f.write("# OCR Prompt Configuration\n")
                f.write(f"# {'Created' if is_new else 'Updated'}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

            action = "created" if is_new else "updated"
            return f"✅ Successfully {action}: {filename}"

        except Exception as e:
            return f"❌ Error saving config: {str(e)}"

    def delete_prompt(self, filename: str) -> str:
        """
        刪除 prompt 配置

        Args:
            filename: 配置文件名

        Returns:
            結果訊息
        """
        if not filename:
            return "❌ Please select a prompt config to delete"

        config_path = self.config_dir / filename

        if not config_path.exists():
            return f"❌ Config file '{filename}' does not exist"

        try:
            config_path.unlink()
            return f"✅ Successfully deleted: {filename}"

        except Exception as e:
            return f"❌ Error deleting config: {str(e)}"

    def get_prompt_table(self) -> List[List[str]]:
        """
        獲取所有 prompt 的表格資料

        Returns:
            表格資料 [[filename, prompt_text], ...]
        """
        prompts = []
        for filename in self.get_prompt_list():
            config_path = self.config_dir / filename
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                prompt_text = config.get('prompt_text', 'N/A')
                prompts.append([filename, prompt_text])
            except:
                prompts.append([filename, "Error loading"])

        return prompts


def create_ui(config_dir: str = "configs/ocr_prompt"):
    """
    創建 Gradio UI

    Args:
        config_dir: 配置文件目錄路徑
    """
    manager = PromptManager(config_dir=config_dir)

    # CSS 樣式
    custom_css = """
    .prompt-table {
        font-size: 14px;
    }
    .success-msg {
        color: green;
        font-weight: bold;
    }
    .error-msg {
        color: red;
        font-weight: bold;
    }
    """

    with gr.Blocks(css=custom_css, title="Prompt Manager") as demo:
        gr.Markdown("# 📝 OCR Prompt Manager")
        gr.Markdown("Manage your OCR prompt configurations")

        with gr.Tabs():
            # Tab 1: 列出所有 Prompts
            with gr.Tab("📋 View All Prompts"):
                refresh_btn = gr.Button("🔄 Refresh", variant="primary")
                prompt_table = gr.Dataframe(
                    headers=["Filename", "Prompt Text"],
                    datatype=["str", "str"],
                    value=manager.get_prompt_table(),
                    interactive=False,
                    elem_classes=["prompt-table"]
                )

                def refresh_table():
                    return manager.get_prompt_table()

                refresh_btn.click(
                    fn=refresh_table,
                    outputs=prompt_table
                )

            # Tab 2: 新增 Prompt
            with gr.Tab("➕ Add New Prompt"):
                gr.Markdown("### Create a new prompt configuration")

                new_filename = gr.Textbox(
                    label="Filename",
                    placeholder="Enter filename (e.g., 3.yml or just 3)",
                    info="Will automatically add .yml extension if not provided"
                )
                new_prompt_text = gr.Textbox(
                    label="Prompt Text",
                    placeholder="Enter your prompt text here...",
                    lines=3
                )
                add_btn = gr.Button("➕ Add Prompt", variant="primary")
                add_output = gr.Textbox(label="Result", interactive=False)

                def add_prompt(filename, prompt_text):
                    result = manager.save_prompt(filename, prompt_text, is_new=True)
                    # 刷新列表
                    table_data = manager.get_prompt_table()
                    return result, table_data

                add_btn.click(
                    fn=add_prompt,
                    inputs=[new_filename, new_prompt_text],
                    outputs=[add_output, prompt_table]
                )

            # Tab 3: 更新 Prompt
            with gr.Tab("✏️ Update Prompt"):
                gr.Markdown("### Update an existing prompt configuration")

                with gr.Row():
                    update_filename = gr.Dropdown(
                        label="Select Prompt to Update",
                        choices=manager.get_prompt_list(),
                        interactive=True
                    )
                    load_btn = gr.Button("📂 Load", size="sm")

                update_prompt_text = gr.Textbox(
                    label="Prompt Text",
                    placeholder="Prompt text will appear here after loading...",
                    lines=3
                )
                update_btn = gr.Button("💾 Update Prompt", variant="primary")
                update_output = gr.Textbox(label="Result", interactive=False)

                def load_selected_prompt(filename):
                    prompt_text, message = manager.load_prompt(filename)
                    return prompt_text, message

                def update_prompt(filename, prompt_text):
                    result = manager.save_prompt(filename, prompt_text, is_new=False)
                    # 刷新列表和下拉選單
                    table_data = manager.get_prompt_table()
                    dropdown_choices = manager.get_prompt_list()
                    return result, table_data, gr.Dropdown(choices=dropdown_choices)

                load_btn.click(
                    fn=load_selected_prompt,
                    inputs=update_filename,
                    outputs=[update_prompt_text, update_output]
                )

                update_btn.click(
                    fn=update_prompt,
                    inputs=[update_filename, update_prompt_text],
                    outputs=[update_output, prompt_table, update_filename]
                )

            # Tab 4: 刪除 Prompt
            with gr.Tab("🗑️ Delete Prompt"):
                gr.Markdown("### Delete a prompt configuration")
                gr.Markdown("⚠️ **Warning**: This action cannot be undone!")

                delete_filename = gr.Dropdown(
                    label="Select Prompt to Delete",
                    choices=manager.get_prompt_list(),
                    interactive=True
                )

                with gr.Row():
                    delete_preview_btn = gr.Button("👁️ Preview", size="sm")
                    delete_btn = gr.Button("🗑️ Delete Prompt", variant="stop")

                delete_preview = gr.Textbox(
                    label="Preview",
                    placeholder="Select a prompt and click Preview to see its content...",
                    interactive=False,
                    lines=2
                )
                delete_output = gr.Textbox(label="Result", interactive=False)

                def preview_delete(filename):
                    prompt_text, message = manager.load_prompt(filename)
                    if prompt_text:
                        return f"File: {filename}\nPrompt: {prompt_text}"
                    return message

                def delete_selected_prompt(filename):
                    result = manager.delete_prompt(filename)
                    # 刷新列表和下拉選單
                    table_data = manager.get_prompt_table()
                    dropdown_choices = manager.get_prompt_list()
                    return result, table_data, gr.Dropdown(choices=dropdown_choices), gr.Dropdown(choices=dropdown_choices), ""

                delete_preview_btn.click(
                    fn=preview_delete,
                    inputs=delete_filename,
                    outputs=delete_preview
                )

                delete_btn.click(
                    fn=delete_selected_prompt,
                    inputs=delete_filename,
                    outputs=[delete_output, prompt_table, delete_filename, update_filename, delete_preview]
                )

        gr.Markdown("---")
        gr.Markdown(
            f"**Config Directory**: `{manager.config_dir.absolute()}`\n\n"
            "**Usage in evaluation**:\n"
            "```bash\n"
            "python core/evaluate.py --prompt-config configs/ocr_prompt/1.yml --model-name your-model\n"
            "```"
        )

    return demo


def main():
    """主程序"""
    import argparse

    parser = argparse.ArgumentParser(description="Prompt Manager Web UI")
    parser.add_argument(
        '--config-dir',
        type=str,
        default='configs/ocr_prompt',
        help='Path to prompt config directory (default: configs/ocr_prompt)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port to run the server on (default: 7860)'
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create a public shareable link'
    )

    args = parser.parse_args()

    # 創建並啟動 UI
    demo = create_ui(config_dir=args.config_dir)
    demo.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0"
    )


if __name__ == "__main__":
    main()
