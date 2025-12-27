#!/usr/bin/env python3
"""
Extract samples from evaluation results that match samples in ocr_test.json.

This script reads an evaluation result JSON file (like gemma3_before_sft_test2700.json)
and extracts only the samples whose image_path exists in ocr_test.json.

Usage:
    python extract_matching_samples.py <evaluation_json> <ocr_test_json> <output_json>

Example:
    python extract_matching_samples.py \
        results/evaluation/test_data=2400/gemma3_before_sft_test2700.json \
        data/benchmark/ocr_test.json \
        output/filtered_gemma3.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Set


def load_json(file_path: str) -> dict:
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: dict, file_path: str) -> None:
    """Save data to JSON file."""
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_image_paths_from_ocr_test(ocr_test_data: List[Dict]) -> Set[str]:
    """Extract all image paths from ocr_test.json."""
    return {sample['image_path'] for sample in ocr_test_data}


def filter_samples(eval_data: dict, valid_image_paths: Set[str]) -> dict:
    """
    Filter evaluation samples to only include those with image_paths in ocr_test.json.

    Args:
        eval_data: Evaluation result dictionary with 'samples' key
        valid_image_paths: Set of valid image paths from ocr_test.json

    Returns:
        Filtered evaluation data dictionary
    """
    original_count = len(eval_data['samples'])

    filtered_samples = [
        sample for sample in eval_data['samples']
        if sample['image_path'] in valid_image_paths
    ]

    filtered_data = eval_data.copy()
    filtered_data['samples'] = filtered_samples
    filtered_data['total_samples'] = len(filtered_samples)

    print(f"Original samples: {original_count}")
    print(f"Filtered samples: {len(filtered_samples)}")
    print(f"Removed samples: {original_count - len(filtered_samples)}")

    return filtered_data


def main():
    parser = argparse.ArgumentParser(
        description='Extract samples from evaluation results that match ocr_test.json'
    )
    parser.add_argument(
        'evaluation_json',
        help='Path to evaluation result JSON (e.g., gemma3_before_sft_test2700.json)'
    )
    parser.add_argument(
        'ocr_test_json',
        help='Path to ocr_test.json benchmark file'
    )
    parser.add_argument(
        'output_json',
        help='Path to output filtered JSON file'
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading evaluation data from: {args.evaluation_json}")
    eval_data = load_json(args.evaluation_json)

    print(f"Loading OCR test data from: {args.ocr_test_json}")
    ocr_test_data = load_json(args.ocr_test_json)

    # Get valid image paths
    valid_image_paths = get_image_paths_from_ocr_test(ocr_test_data)
    print(f"OCR test contains {len(valid_image_paths)} unique image paths")

    # Filter samples
    filtered_data = filter_samples(eval_data, valid_image_paths)

    # Save result
    print(f"Saving filtered data to: {args.output_json}")
    save_json(filtered_data, args.output_json)
    print("Done!")


if __name__ == '__main__':
    main()
