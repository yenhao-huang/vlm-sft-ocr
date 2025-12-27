"""
Add focus_layout information to evaluation results.

This script loads evaluation results and enriches them with focus_layout information
based on matching image paths.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any


def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_layout_index(layout_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Build an index of focus_layout data by image_path."""
    layout_index = {}
    for result in layout_data.get('results', []):
        image_path = result.get('image_path')
        if image_path:
            layout_index[image_path] = result.get('focus_layout', [])
    return layout_index


def add_layout_to_samples(
    eval_data: Dict[str, Any],
    layout_index: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Add focus_layout information to evaluation samples."""
    enriched_data = eval_data.copy()

    # Statistics
    matched_count = 0
    total_count = len(enriched_data.get('samples', []))

    # Add focus_layout to each sample
    for sample in enriched_data.get('samples', []):
        image_path = sample.get('image_path')
        if image_path and image_path in layout_index:
            sample['focus_layout'] = layout_index[image_path]
            matched_count += 1
        else:
            sample['focus_layout'] = []

    # Add metadata
    enriched_data['layout_metadata'] = {
        'total_samples': total_count,
        'matched_samples': matched_count,
        'unmatched_samples': total_count - matched_count,
        'match_rate': matched_count / total_count if total_count > 0 else 0
    }

    return enriched_data, matched_count, total_count


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Add focus_layout information to evaluation results.'
    )
    parser.add_argument(
        '--layout-file',
        type=str,
        default='data/layout/layout_results_postprocessed_20251223_134641.json',
        help='Path to layout results JSON file'
    )
    parser.add_argument(
        '--source-dir',
        type=str,
        default='results/evaluation/test_data=2400',
        help='Directory containing evaluation JSON files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/evaluation/test_data=2400/layout',
        help='Output directory for enriched JSON files'
    )
    return parser.parse_args()


def main():
    # Parse configuration
    args = parse_args()
    layout_file = args.layout_file
    source_dir = args.source_dir
    output_dir = args.output_dir

    # Load layout data
    print(f"Loading layout data from: {layout_file}")
    layout_data = load_json(layout_file)
    layout_index = build_layout_index(layout_data)
    print(f"Loaded focus_layout information for {len(layout_index)} images")

    # Process all JSON files in source directory
    source_path = Path(source_dir)
    json_files = list(source_path.glob('*.json'))

    print(f"\nFound {len(json_files)} JSON files to process")

    for json_file in json_files:
        print(f"\nProcessing: {json_file.name}")

        # Load evaluation data
        eval_data = load_json(str(json_file))

        # Add focus_layout information
        enriched_data, matched, total = add_layout_to_samples(eval_data, layout_index)

        # Save enriched data
        output_file = os.path.join(output_dir, json_file.name)
        save_json(enriched_data, output_file)

        print(f"  ✓ Matched {matched}/{total} samples ({matched/total*100:.1f}%)")
        print(f"  ✓ Saved to: {output_file}")

    print(f"\n{'='*60}")
    print(f"All files processed successfully!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
