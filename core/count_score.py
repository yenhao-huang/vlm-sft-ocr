#!/usr/bin/env python3
"""
Recalculate CER and F1 scores with normalized text (whitespace and punctuation removed)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from datetime import datetime
import re
import string
from jiwer import cer
from lib.utils.count_metric import batch_char_f1_score, char_f1_score


def normalize_text(text):
    """
    Normalize text by removing whitespace and punctuation

    Args:
        text: Input text string

    Returns:
        Normalized text with whitespace and punctuation removed
    """
    # Remove all whitespace (spaces, tabs, newlines, etc.)
    text = re.sub(r'\s+', '', text)

    # Remove all punctuation (both ASCII and Chinese punctuation)
    # ASCII punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Common Chinese punctuation marks
    chinese_punctuation = '，。！？；：""''『』「」（）【】《》〈〉、·…—～︰︱︳︴︵︶︷︸︹︺︻︼︽︾︿﹀﹁﹂﹃﹄﹙﹚﹛﹜﹝﹞'
    text = text.translate(str.maketrans('', '', chinese_punctuation))

    return text


def count_score(results):
    """
    Calculate CER and F1 scores from results

    Args:
        results: List of dictionaries containing prediction, ground_truth, and image_path

    Returns:
        Dictionary containing overall and individual scores
    """

    predictions = [res["prediction"] for res in results]
    references = [res["ground_truth"] for res in results]

    # Normalize text by removing whitespace and punctuation
    predictions_normalized = [normalize_text(pred) for pred in predictions]
    references_normalized = [normalize_text(ref) for ref in references]

    # Calculate scores using normalized text
    cer_score = cer(references_normalized, predictions_normalized)
    f1 = batch_char_f1_score(predictions_normalized, references_normalized)

    individual_cer_scores = [cer([ref], [pred]) for ref, pred in zip(references_normalized, predictions_normalized)]
    individual_f1_scores = [char_f1_score(pred, ref) for pred, ref in zip(predictions_normalized, references_normalized)]

    return {
        "overall_cer": cer_score,
        "overall_f1": f1,
        "individual_cer_scores": individual_cer_scores,
        "individual_f1_scores": individual_f1_scores,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Recalculate CER and F1 scores with text normalization'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSON file path (e.g., results/gemma-12b-base.json)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSON file path (e.g., results/gemma-12b-base_norm.json)'
    )
    args = parser.parse_args()

    # Read input JSON
    print(f"Reading input file: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Prepare results for count_score function
    results = []
    for sample in data['samples']:
        results.append({
            'prediction': sample['prediction'],
            'ground_truth': sample['label'],
            'image_path': sample['image_path']
        })

    # Calculate normalized scores
    print("Calculating normalized scores (removing whitespace and punctuation)...")
    scores = count_score(results)

    # Update data with new scores
    data['overall_cer'] = scores['overall_cer']
    data['overall_f1'] = scores['overall_f1']
    data['timestamp'] = datetime.now().isoformat()

    # Update individual sample scores
    for i, sample in enumerate(data['samples']):
        sample['cer_score'] = scores['individual_cer_scores'][i]
        sample['f1_score'] = scores['individual_f1_scores'][i]

    # Write output JSON
    print(f"Writing output file: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("Score Comparison:")
    print("=" * 60)

    # Read original scores for comparison
    with open(args.input, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    print(f"Original Overall CER: {original_data['overall_cer']:.6f}")
    print(f"Normalized Overall CER: {data['overall_cer']:.6f}")
    print(f"Improvement: {((original_data['overall_cer'] - data['overall_cer']) / original_data['overall_cer'] * 100):.2f}%")
    print("-" * 60)
    print(f"Original Overall F1: {original_data['overall_f1']:.6f}")
    print(f"Normalized Overall F1: {data['overall_f1']:.6f}")
    print(f"Improvement: {((data['overall_f1'] - original_data['overall_f1']) / original_data['overall_f1'] * 100):.2f}%")
    print("=" * 60)
    print(f"\nDone! Results saved to {args.output}")


if __name__ == '__main__':
    main()
