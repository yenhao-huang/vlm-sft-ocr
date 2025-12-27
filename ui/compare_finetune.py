import json
import gradio as gr
from PIL import Image
import os
import argparse

def load_results(model1_path, model2_path):
    """Load evaluation results from JSON files"""
    with open(model1_path, 'r', encoding='utf-8') as f:
        model1_results = json.load(f)
    with open(model2_path, 'r', encoding='utf-8') as f:
        model2_results = json.load(f)
    return model1_results, model2_results

def format_metrics(sample):
    """Format metrics for display"""
    return f"CER: {sample['cer_score']:.4f}\nF1: {sample['f1_score']:.4f}"

def format_text_with_metrics(prediction, cer, f1):
    """Format prediction text with metrics"""
    metrics = f"CER: {cer:.4f} | F1: {f1:.4f}\n\n"
    return metrics + prediction

def create_comparison_ui(model1_path, model2_path):
    # Load results
    model1_results, model2_results = load_results(model1_path, model2_path)

    # Create mapping from image_path to sample for model2
    model2_map = {sample['image_path']: sample for sample in model2_results['samples']}

    # Filter model1 samples to only include those with matching image_path in model2
    matched_samples = []
    for m1_sample in model1_results['samples']:
        if m1_sample['image_path'] in model2_map:
            matched_samples.append((m1_sample, model2_map[m1_sample['image_path']]))

    # Get sample count
    num_samples = len(matched_samples)

    def display_sample(sample_idx):
        """Display a specific sample"""
        if sample_idx < 0 or sample_idx >= num_samples:
            return None, "", "Invalid index", "Invalid index", "Invalid index"

        model1_sample, model2_sample = matched_samples[sample_idx]

        # Load image
        image_path = model1_sample['image_path']
        if os.path.exists(image_path):
            img = Image.open(image_path)
        else:
            img = None

        # Format outputs
        model1_text = format_text_with_metrics(
            model1_sample['prediction'],
            model1_sample['cer_score'],
            model1_sample['f1_score']
        )

        model2_text = format_text_with_metrics(
            model2_sample['prediction'],
            model2_sample['cer_score'],
            model2_sample['f1_score']
        )

        label_text = model1_sample['label']

        return img, image_path, label_text, model1_text, model2_text

    # Create Gradio interface
    with gr.Blocks(title="Model Comparison") as demo:
        gr.Markdown("# OCR Model Comparison")
        gr.Markdown(f"**Model 1:** {model1_results['model_name']}")
        gr.Markdown(f"**Overall Metrics (Model 1):** CER={model1_results['overall_cer']:.4f}, F1={model1_results['overall_f1']:.4f}")
        gr.Markdown(f"**Model 2:** {model2_results['model_name']}")
        gr.Markdown(f"**Overall Metrics (Model 2):** CER={model2_results['overall_cer']:.4f}, F1={model2_results['overall_f1']:.4f}")
        gr.Markdown(f"**Total Samples:** {num_samples}")

        # Current index state
        current_idx = gr.State(value=0)

        with gr.Row():
            prev_btn = gr.Button("← 上一個")
            sample_info = gr.Markdown(f"<center>樣本 1 / {num_samples}</center>")
            next_btn = gr.Button("下一個 →")

        with gr.Row():
            with gr.Column(scale=2):
                idx_input = gr.Number(label="跳轉到樣本編號", value=1, precision=0, minimum=1, maximum=num_samples)
            with gr.Column(scale=1):
                jump_btn = gr.Button("跳轉", variant="primary")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Image")
                image_display = gr.Image(label="Input Image", type="pil", height=600)
                image_path_display = gr.Textbox(label="Image Path", lines=1, interactive=False)
                gr.Markdown("### Ground Truth")
                label_display = gr.Textbox(label="", lines=15, max_lines=20)

            with gr.Column(scale=1):
                gr.Markdown("### Model 1")
                model1_output = gr.Textbox(label="", lines=15, max_lines=20)

            with gr.Column(scale=1):
                gr.Markdown("### Model 2")
                model2_output = gr.Textbox(label="", lines=15, max_lines=20)

        # Navigation functions
        def go_prev(idx):
            new_idx = max(0, idx - 1)
            img, img_path, label, model1, model2 = display_sample(new_idx)
            info = f"<center>樣本 {new_idx + 1} / {num_samples}</center>"
            return new_idx, info, img, img_path, label, model1, model2

        def go_next(idx):
            new_idx = min(num_samples - 1, idx + 1)
            img, img_path, label, model1, model2 = display_sample(new_idx)
            info = f"<center>樣本 {new_idx + 1} / {num_samples}</center>"
            return new_idx, info, img, img_path, label, model1, model2

        def jump_to_idx(target_idx):
            # Convert 1-based to 0-based index
            new_idx = max(0, min(num_samples - 1, int(target_idx) - 1))
            img, img_path, label, model1, model2 = display_sample(new_idx)
            info = f"<center>樣本 {new_idx + 1} / {num_samples}</center>"
            return new_idx, info, img, img_path, label, model1, model2

        # Button click events
        prev_btn.click(
            fn=go_prev,
            inputs=[current_idx],
            outputs=[current_idx, sample_info, image_display, image_path_display, label_display, model1_output, model2_output]
        )

        next_btn.click(
            fn=go_next,
            inputs=[current_idx],
            outputs=[current_idx, sample_info, image_display, image_path_display, label_display, model1_output, model2_output]
        )

        jump_btn.click(
            fn=jump_to_idx,
            inputs=[idx_input],
            outputs=[current_idx, sample_info, image_display, image_path_display, label_display, model1_output, model2_output]
        )

        # Load initial sample
        demo.load(
            fn=lambda: (0, f"<center>樣本 1 / {num_samples}</center>") + display_sample(0),
            inputs=[],
            outputs=[current_idx, sample_info, image_display, image_path_display, label_display, model1_output, model2_output]
        )

    return demo

if __name__ == "__main__":
    '''
    ython ui/compare_finetune.py --model1 results/evaluation/test_data\=2400/gemma3_before_sft_test2700.json --model2 results/evaluation/test_data\=2400/gemma3_sft_lr2e4_ep5_test2700.json
    '''
    parser = argparse.ArgumentParser(description='Compare two OCR model evaluation results')
    parser.add_argument('--model1', type=str, required=True, help='Path to first model evaluation JSON')
    parser.add_argument('--model2', type=str, required=True, help='Path to second model evaluation JSON')
    parser.add_argument('--port', type=int, default=7863, help='Server port (default: 7863)')
    args = parser.parse_args()

    demo = create_comparison_ui(args.model1, args.model2)
    demo.launch(share=False, server_name="0.0.0.0", server_port=args.port)
