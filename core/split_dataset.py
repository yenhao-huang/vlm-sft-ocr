import json
import random
import argparse
from pathlib import Path


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Split OCR dataset into train and test sets')
    parser.add_argument('--input', type=str, default='data/raw_data/ocr.json',
                        help='Path to input JSON file (default: data/raw_data/ocr.json)')
    parser.add_argument('--output_dir', type=str, default='data/input',
                        help='Output directory for split datasets (default: data/input)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--test_ratio', type=float, default=1,
                        help='Ratio of test to non_test data (default: 1.0, meaning 1:1)')
    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Read the original data
    input_file = Path(args.input)
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Total items: {len(data)}")

    # Shuffle the data
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    # Calculate split sizes based on test_ratio (non_test:test)
    # For example, ratio 5.0 means 1:(5.0) => non_test = 1/6 of total, test = 5/6 of total
    total = len(shuffled_data)
    non_test_size = total // int(args.test_ratio + 1)
    test_size = total - non_test_size

    print(f"non_test size: {non_test_size}")
    print(f"test size: {test_size}")
    print(f"Ratio: 1:{test_size/non_test_size:.2f}")

    # Split the data and keep only img_path and ocr_results
    non_test_data = [
        {"image_path": item["img_path"], "ocr_text": item["ocr_results"]}
        for item in shuffled_data[:non_test_size]
    ]
    test_data = [
        {"image_path": item["img_path"], "ocr_text": item["ocr_results"]}
        for item in shuffled_data[non_test_size:]
    ]

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write non_test dataset
    non_test_file = output_dir / "ocr_non_test.json"
    with open(non_test_file, 'w', encoding='utf-8') as f:
        json.dump(non_test_data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(non_test_data)} items to {non_test_file}")

    # Write test dataset
    test_file = output_dir / "ocr_test.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(test_data)} items to {test_file}")

    print("\nSplit complete!")


if __name__ == "__main__":
    main()
