import gradio as gr
import json
import os
from PIL import Image
from pathlib import Path


def get_evaluation_files(eval_dir="results/evaluation"):
    """
    Get all JSON files in the evaluation directory.
    """
    eval_path = Path(eval_dir)
    if not eval_path.exists():
        return []

    json_files = sorted(eval_path.glob("*.json"))
    return [str(f) for f in json_files]


def load_results(json_file):
    """
    Load evaluation results from JSON file.
    """
    if not os.path.exists(json_file):
        return None, f"File not found: {json_file}"

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        return results, None
    except Exception as e:
        return None, f"Error loading file: {str(e)}"


def create_viewer(eval_dir="results/evaluation", initial_file=None):
    """
    Create Gradio interface for viewing evaluation results.
    """
    # Get all evaluation files
    available_files = get_evaluation_files(eval_dir)

    if not available_files:
        print(f"No JSON files found in {eval_dir}")
        return None

    # Set initial file
    if initial_file is None or initial_file not in available_files:
        initial_file = available_files[0]

    # Create Gradio interface
    with gr.Blocks(title="OCR Evaluation Viewer") as demo:
        gr.Markdown("# OCR Evaluation Results Viewer")

        # File selector
        with gr.Row():
            file_dropdown = gr.Dropdown(
                choices=available_files,
                value=initial_file,
                label="Select Evaluation File",
                scale=3
            )
            refresh_btn = gr.Button("🔄 Refresh Files", variant="secondary", scale=1)

        # Stats display
        stats_display = gr.Markdown()

        # State variables
        current_results = gr.State(value=None)
        current_samples = gr.State(value=[])
        current_idx = gr.State(value=0)

        def load_file(json_file):
            """Load a new file and return updated UI components."""
            results, error = load_results(json_file)

            if error:
                error_msg = f"## Error\n{error}"
                return results, [], 0, error_msg, None, "", "No label", "No prediction", 1, 0

            samples = results.get("samples", [])
            total_samples = len(samples)

            if total_samples == 0:
                error_msg = "## No samples found in results file"
                return results, samples, 0, error_msg, None, "", "No label", "No prediction", 1, 0

            # Generate stats markdown
            stats_md = f"""
**Model:** {results.get('model_name', 'N/A')}
**Overall CER:** {results.get('overall_cer', 0.0):.4f} ({results.get('overall_cer', 0.0)*100:.2f}%)
**Overall F1:** {results.get('overall_f1', 0.0):.4f} ({results.get('overall_f1', 0.0)*100:.2f}%)
**Total Samples:** {total_samples}
**Timestamp:** {results.get('timestamp', 'N/A')}
            """

            # Load first sample
            sample = samples[0]
            image_path = sample.get("image_path", "")
            label = sample.get("label", "")
            prediction = sample.get("prediction", "")

            try:
                if os.path.exists(image_path):
                    img = Image.open(image_path)
                else:
                    img = None
            except Exception:
                img = None

            return results, samples, 0, stats_md, img, image_path, label, prediction, total_samples, 0

        def get_sample_data(samples_list, sample_idx):
            """Get data for a specific sample index."""
            if not samples_list or sample_idx < 0 or sample_idx >= len(samples_list):
                return None, "", "No label", "No prediction"

            sample = samples_list[sample_idx]
            image_path = sample.get("image_path", "")
            label = sample.get("label", "")
            prediction = sample.get("prediction", "")

            try:
                if os.path.exists(image_path):
                    img = Image.open(image_path)
                else:
                    img = None
            except Exception:
                img = None

            return img, image_path, label, prediction

        def next_sample(samples_list, current_idx_val):
            """Go to next sample."""
            if not samples_list:
                return current_idx_val, None, "", "No label", "No prediction"

            next_idx = (current_idx_val + 1) % len(samples_list)
            return next_idx, *get_sample_data(samples_list, next_idx)

        def prev_sample(samples_list, current_idx_val):
            """Go to previous sample."""
            if not samples_list:
                return current_idx_val, None, "", "No label", "No prediction"

            prev_idx = (current_idx_val - 1) % len(samples_list)
            return prev_idx, *get_sample_data(samples_list, prev_idx)

        def goto_sample(samples_list, sample_num):
            """Go to specific sample number (1-indexed)."""
            if not samples_list:
                return 0, None, "", "No label", "No prediction"

            idx = max(0, min(sample_num - 1, len(samples_list) - 1))
            return idx, *get_sample_data(samples_list, idx)

        def refresh_files(current_file):
            """Refresh the file list."""
            new_files = get_evaluation_files(eval_dir)
            if not new_files:
                return gr.update(choices=[]), current_file

            # Keep current file if it still exists, otherwise select first
            new_value = current_file if current_file in new_files else new_files[0]
            return gr.update(choices=new_files, value=new_value), new_value

        # Navigation controls
        with gr.Row():
            prev_btn = gr.Button("⬅️ Previous", variant="secondary", scale=1)
            next_btn = gr.Button("Next ➡️", variant="primary", scale=1)
            goto_input = gr.Number(label="Go to Sample", value=1, minimum=1, step=1, scale=1)
            goto_btn = gr.Button("Go", variant="secondary", scale=1)

        # Main content: Image, Ground Truth, and Prediction in one row
        with gr.Row():
            with gr.Column(scale=2):
                # Image display - larger
                image_display = gr.Image(label="Image", type="pil", height=800)
                # Image path display
                image_path_display = gr.Textbox(label="Image Path", lines=1, interactive=False)

            with gr.Column(scale=1):
                label_display = gr.Textbox(label="Ground Truth", lines=35, max_lines=40, interactive=False)

            with gr.Column(scale=1):
                pred_display = gr.Textbox(label="Prediction", lines=35, max_lines=40, interactive=False)

        # Event handlers
        file_dropdown.change(
            fn=load_file,
            inputs=[file_dropdown],
            outputs=[
                current_results,
                current_samples,
                current_idx,
                stats_display,
                image_display,
                image_path_display,
                label_display,
                pred_display,
                goto_input,
                goto_input
            ]
        )

        refresh_btn.click(
            fn=refresh_files,
            inputs=[file_dropdown],
            outputs=[file_dropdown, file_dropdown]
        )

        next_btn.click(
            fn=next_sample,
            inputs=[current_samples, current_idx],
            outputs=[current_idx, image_display, image_path_display, label_display, pred_display]
        )

        prev_btn.click(
            fn=prev_sample,
            inputs=[current_samples, current_idx],
            outputs=[current_idx, image_display, image_path_display, label_display, pred_display]
        )

        goto_btn.click(
            fn=goto_sample,
            inputs=[current_samples, goto_input],
            outputs=[current_idx, image_display, image_path_display, label_display, pred_display]
        )

        # Load first file on startup
        demo.load(
            fn=load_file,
            inputs=[file_dropdown],
            outputs=[
                current_results,
                current_samples,
                current_idx,
                stats_display,
                image_display,
                image_path_display,
                label_display,
                pred_display,
                goto_input,
                goto_input
            ]
        )

    return demo


def main(
    eval_dir="results/evaluation",
    initial_file=None,
    share=False,
    server_name="0.0.0.0",
    server_port=7860
):
    """
    Launch the evaluation viewer.
    """
    demo = create_viewer(eval_dir, initial_file)

    if demo is None:
        print("Failed to create viewer")
        return

    print(f"\nLaunching viewer for directory: {eval_dir}")
    demo.launch(
        share=share,
        server_name=server_name,
        server_port=server_port
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OCR Evaluation Results Viewer")
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="results/evaluation/test_data=2400",
        help="Directory containing evaluation result JSON files (default: results/evaluation)"
    )
    parser.add_argument(
        "--initial_file",
        type=str,
        default=None,
        help="Initial file to load (optional)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link"
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default="0.0.0.0",
        help="Server name (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=7860,
        help="Server port (default: 7860)"
    )

    args = parser.parse_args()

    main(
        eval_dir=args.eval_dir,
        initial_file=args.initial_file,
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port
    )
