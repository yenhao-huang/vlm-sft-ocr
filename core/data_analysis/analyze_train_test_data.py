import json
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path


# Load datasets
def load_data():
    train_path = "data/input/ocr_non_test_data=500.json"
    test_path = "data/input/ocr_test_data=100.json"

    with open(train_path, 'r', encoding='utf-8') as f:
        train_set = json.load(f)

    with open(test_path, 'r', encoding='utf-8') as f:
        test_set = json.load(f)

    return train_set, test_set


def content_overlap():
    """
    Analyze overlap between train and test sets
    Check how many pages and PDFs from test set appear in train set
    """
    train_set, test_set = load_data()

    # Extract page and pdf info from train set
    train_pages = set()
    train_pdfs = set()

    for item in train_set:
        image_path = item["image_path"]
        pdf = image_path.split("/")[-2]
        page = image_path.split("/")[-1]
        train_pages.add((pdf, page))
        train_pdfs.add(pdf)

    # Check overlap in test set
    page_overlap_count = 0
    pdf_overlap_count = 0
    overlapping_pages = []
    overlapping_pdfs = []

    for item in test_set:
        image_path = item["image_path"]
        pdf = image_path.split("/")[-2]
        page = image_path.split("/")[-1]

        # Check if same page appears in train set
        if (pdf, page) in train_pages:
            page_overlap_count += 1
            overlapping_pages.append((pdf, page))

        # Check if same pdf appears in train set
        if pdf in train_pdfs:
            pdf_overlap_count += 1
            overlapping_pdfs.append(pdf)

    # Print results
    print("=" * 80)
    print("Content Overlap Analysis")
    print("=" * 80)
    print(f"Total test samples: {len(test_set)}")
    print(f"Pages from test set appearing in train set: {page_overlap_count}")
    print(f"PDFs from test set appearing in train set: {pdf_overlap_count}")
    print(f"\nOverlap ratio (pages): {page_overlap_count / len(test_set) * 100:.2f}%")
    print(f"Overlap ratio (PDFs): {pdf_overlap_count / len(test_set) * 100:.2f}%")

    if overlapping_pages:
        print(f"\nFirst 5 overlapping pages:")
        for pdf, page in overlapping_pages[:5]:
            print(f"  PDF: {pdf}, Page: {page}")

    return {
        'page_overlap_count': page_overlap_count,
        'pdf_overlap_count': pdf_overlap_count,
        'total_test_samples': len(test_set),
        'overlapping_pages': overlapping_pages,
        'overlapping_pdfs': overlapping_pdfs
    }


def get_pdf_distribution():
    """
    Analyze and plot PDF distribution for both train and test sets
    """
    train_set, test_set = load_data()

    # Extract PDF names
    train_pdfs = [item["image_path"].split("/")[-2] for item in train_set]
    test_pdfs = [item["image_path"].split("/")[-2] for item in test_set]

    # Count distribution
    train_pdf_dist = Counter(train_pdfs)
    test_pdf_dist = Counter(test_pdfs)

    # Create output directory
    output_dir = Path("core/data_analysis/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot train set distribution
    plt.figure(figsize=(15, 8))
    pdfs = list(train_pdf_dist.keys())
    counts = list(train_pdf_dist.values())
    plt.bar(range(len(pdfs)), counts)
    plt.xlabel('PDF Index')
    plt.ylabel('Number of Pages')
    plt.title(f'Train Set PDF Distribution (Total PDFs: {len(pdfs)})')
    plt.xticks(range(len(pdfs)), range(len(pdfs)), rotation=90)
    plt.tight_layout()
    plt.savefig(output_dir / 'train_pdf_distribution.png', dpi=150)
    print(f"Saved: {output_dir / 'train_pdf_distribution.png'}")
    plt.close()

    # Plot test set distribution
    plt.figure(figsize=(15, 8))
    pdfs = list(test_pdf_dist.keys())
    counts = list(test_pdf_dist.values())
    plt.bar(range(len(pdfs)), counts)
    plt.xlabel('PDF Index')
    plt.ylabel('Number of Pages')
    plt.title(f'Test Set PDF Distribution (Total PDFs: {len(pdfs)})')
    plt.xticks(range(len(pdfs)), range(len(pdfs)), rotation=90)
    plt.tight_layout()
    plt.savefig(output_dir / 'test_pdf_distribution.png', dpi=150)
    print(f"Saved: {output_dir / 'test_pdf_distribution.png'}")
    plt.close()

    # Print statistics
    print("\n" + "=" * 80)
    print("PDF Distribution Analysis")
    print("=" * 80)
    print(f"\nTrain Set:")
    print(f"  Total samples: {len(train_set)}")
    print(f"  Unique PDFs: {len(train_pdf_dist)}")
    print(f"  Avg pages per PDF: {len(train_set) / len(train_pdf_dist):.2f}")
    print(f"  Max pages from one PDF: {max(train_pdf_dist.values())}")
    print(f"  Min pages from one PDF: {min(train_pdf_dist.values())}")

    print(f"\nTest Set:")
    print(f"  Total samples: {len(test_set)}")
    print(f"  Unique PDFs: {len(test_pdf_dist)}")
    print(f"  Avg pages per PDF: {len(test_set) / len(test_pdf_dist):.2f}")
    print(f"  Max pages from one PDF: {max(test_pdf_dist.values())}")
    print(f"  Min pages from one PDF: {min(test_pdf_dist.values())}")

    # Top 10 PDFs
    print(f"\nTop 10 PDFs in Train Set:")
    for pdf, count in train_pdf_dist.most_common(10):
        print(f"  {pdf}: {count} pages")

    print(f"\nTop 10 PDFs in Test Set:")
    for pdf, count in test_pdf_dist.most_common(10):
        print(f"  {pdf}: {count} pages")

    return train_pdf_dist, test_pdf_dist


def ocr_text_len():
    """
    Analyze OCR text length distribution for train and test sets
    """
    train_set, test_set = load_data()

    # Calculate text lengths
    train_lengths = [len(item["ocr_text"]) for item in train_set]
    test_lengths = [len(item["ocr_text"]) for item in test_set]

    # Create output directory
    output_dir = Path("core/data_analysis/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot histograms
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Train set histogram
    axes[0].hist(train_lengths, bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Text Length (characters)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Train Set OCR Text Length Distribution')
    axes[0].grid(True, alpha=0.3)

    # Test set histogram
    axes[1].hist(test_lengths, bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Text Length (characters)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Test Set OCR Text Length Distribution')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'ocr_text_length_distribution.png', dpi=150)
    print(f"Saved: {output_dir / 'ocr_text_length_distribution.png'}")
    plt.close()

    # Box plot comparison
    plt.figure(figsize=(10, 6))
    plt.boxplot([train_lengths, test_lengths], labels=['Train Set', 'Test Set'])
    plt.ylabel('Text Length (characters)')
    plt.title('OCR Text Length Comparison')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'ocr_text_length_boxplot.png', dpi=150)
    print(f"Saved: {output_dir / 'ocr_text_length_boxplot.png'}")
    plt.close()

    # Print statistics
    print("\n" + "=" * 80)
    print("OCR Text Length Analysis")
    print("=" * 80)
    print(f"\nTrain Set:")
    print(f"  Mean length: {sum(train_lengths) / len(train_lengths):.2f} characters")
    print(f"  Median length: {sorted(train_lengths)[len(train_lengths) // 2]:.2f} characters")
    print(f"  Max length: {max(train_lengths)} characters")
    print(f"  Min length: {min(train_lengths)} characters")

    print(f"\nTest Set:")
    print(f"  Mean length: {sum(test_lengths) / len(test_lengths):.2f} characters")
    print(f"  Median length: {sorted(test_lengths)[len(test_lengths) // 2]:.2f} characters")
    print(f"  Max length: {max(test_lengths)} characters")
    print(f"  Min length: {min(test_lengths)} characters")

    return train_lengths, test_lengths


def main():
    """
    Run all analysis functions
    """
    print("Starting data analysis...\n")

    # Analysis 1: Content overlap
    content_overlap()

    # Analysis 2: PDF distribution
    get_pdf_distribution()

    # Analysis 3: OCR text length
    ocr_text_len()

    print("\n" + "=" * 80)
    print("All analyses completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
