from collections import Counter


def char_f1_score(pred, gt):
    """
    Calculate character-level F1 score between prediction and ground truth.
    This is used for normalized text where whitespace is removed.
    """
    # Convert strings to character lists
    pred_chars = list(pred)
    gt_chars = list(gt)

    # Count character occurrences
    common = Counter(pred_chars) & Counter(gt_chars)
    tp = sum(common.values())

    if tp == 0:
        return 0.0

    if len(pred_chars) == 0 or len(gt_chars) == 0:
        return 0.0

    precision = tp / len(pred_chars)
    recall = tp / len(gt_chars)
    return 2 * precision * recall / (precision + recall)


def batch_char_f1_score(predictions, references):
    """
    Calculate average character-level F1 score for a batch of predictions.
    """
    if len(predictions) == 0:
        return 0.0
    return sum(char_f1_score(pred, ref) for pred, ref in zip(predictions, references)) / len(predictions)


def f1_score(pred, gt):
    """
    Calculate F1 score between prediction and ground truth.
    """
    pred_tokens = pred.split()
    gt_tokens = gt.split()

    common = Counter(pred_tokens) & Counter(gt_tokens)
    tp = sum(common.values())

    if tp == 0:
        return 0.0

    precision = tp / len(pred_tokens)
    recall = tp / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def batch_f1_score(references, predictions):
    """
    Calculate average F1 score for a batch of predictions.
    """
    return sum(f1_score(pred, ref) for pred, ref in zip(predictions, references)) / len(predictions)

