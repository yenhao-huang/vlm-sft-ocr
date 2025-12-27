import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from core.count_score import normalize_text
from lib.utils.count_metric import char_f1_score


class TestNormalizeText(unittest.TestCase):
    """Test cases for the normalize_text function"""

    def test_remove_spaces(self):
        """Test removing spaces"""
        input_text = "Hello World Test"
        expected = "HelloWorldTest"
        result = normalize_text(input_text)
        self.assertEqual(result, expected)

    def test_remove_newlines(self):
        """Test removing newlines"""
        input_text = "Line 1\nLine 2\nLine 3"
        expected = "Line1Line2Line3"
        result = normalize_text(input_text)
        self.assertEqual(result, expected)

    def test_remove_tabs(self):
        """Test removing tabs"""
        input_text = "Column1\tColumn2\tColumn3"
        expected = "Column1Column2Column3"
        result = normalize_text(input_text)
        self.assertEqual(result, expected)

    def test_remove_ascii_punctuation(self):
        """Test removing ASCII punctuation marks"""
        input_text = "Hello, World! How are you? I'm fine."
        expected = "HelloWorldHowareyouImfine"
        result = normalize_text(input_text)
        self.assertEqual(result, expected)

    def test_remove_chinese_punctuation(self):
        """Test removing Chinese punctuation marks"""
        input_text = "你好，世界！這是測試。"
        expected = "你好世界這是測試"
        result = normalize_text(input_text)
        self.assertEqual(result, expected)

    def test_remove_mixed_punctuation(self):
        """Test removing both ASCII and Chinese punctuation"""
        input_text = "Hello，世界！Test：測試（example）"
        expected = "Hello世界Test測試example"
        result = normalize_text(input_text)
        self.assertEqual(result, expected)

    def test_complex_chinese_text(self):
        """Test with complex Chinese text including various punctuation"""
        input_text = "檔 號：\n保存年限：\n衛生福利部 令"
        expected = "檔號保存年限衛生福利部令"
        result = normalize_text(input_text)
        self.assertEqual(result, expected)

    def test_ocr_sample_text(self):
        """Test with sample OCR text from the dataset"""
        input_text = "發文日期：中華民國112年5月10日\n發文字號：衛部醫字第1121663253號"
        expected = "發文日期中華民國112年5月10日發文字號衛部醫字第1121663253號"
        result = normalize_text(input_text)
        self.assertEqual(result, expected)

    def test_special_chinese_punctuation(self):
        """Test removing special Chinese punctuation marks"""
        input_text = "《書名》、【注釋】、「引用」、''單引號''"
        expected = "書名注釋引用單引號"
        result = normalize_text(input_text)
        self.assertEqual(result, expected)

    def test_mixed_whitespace(self):
        """Test removing mixed whitespace characters"""
        input_text = "Text  with   multiple\t\tspaces\n\nand\r\nnewlines"
        expected = "Textwithmultiplespacesandnewlines"
        result = normalize_text(input_text)
        self.assertEqual(result, expected)

    def test_ellipsis_and_dash(self):
        """Test removing ellipsis and dashes"""
        input_text = "測試…省略號—破折號～波浪號"
        expected = "測試省略號破折號波浪號"
        result = normalize_text(input_text)
        self.assertEqual(result, expected)

    def test_empty_string(self):
        """Test with empty string"""
        input_text = ""
        expected = ""
        result = normalize_text(input_text)
        self.assertEqual(result, expected)

    def test_only_whitespace(self):
        """Test with string containing only whitespace"""
        input_text = "   \t\n\r  "
        expected = ""
        result = normalize_text(input_text)
        self.assertEqual(result, expected)

    def test_only_punctuation(self):
        """Test with string containing only punctuation"""
        input_text = ",.!?;:，。！？；："
        expected = ""
        result = normalize_text(input_text)
        self.assertEqual(result, expected)

    def test_alphanumeric_preserved(self):
        """Test that alphanumeric characters are preserved"""
        input_text = "ABC123xyz456中文789測試"
        expected = "ABC123xyz456中文789測試"
        result = normalize_text(input_text)
        self.assertEqual(result, expected)

    def test_parentheses_and_brackets(self):
        """Test removing various types of brackets"""
        input_text = "(text1) [text2] {text3} （中文1） 【中文2】"
        expected = "text1text2text3中文1中文2"
        result = normalize_text(input_text)
        self.assertEqual(result, expected)


class TestCharF1Score(unittest.TestCase):
    """Test cases for the char_f1_score function"""

    def test_identical_strings(self):
        """Test F1 score with identical strings"""
        pred = "HelloWorld"
        ref = "HelloWorld"
        score = char_f1_score(pred, ref)
        self.assertEqual(score, 1.0)

    def test_completely_different_strings(self):
        """Test F1 score with completely different strings"""
        pred = "ABC"
        ref = "XYZ"
        score = char_f1_score(pred, ref)
        self.assertEqual(score, 0.0)

    def test_partial_match(self):
        """Test F1 score with partial match"""
        pred = "ABCD"
        ref = "BCDE"
        score = char_f1_score(pred, ref)
        # Common chars: B, C, D (3 chars)
        # Precision: 3/4 = 0.75
        # Recall: 3/4 = 0.75
        # F1: 2 * 0.75 * 0.75 / (0.75 + 0.75) = 0.75
        self.assertAlmostEqual(score, 0.75, places=5)

    def test_chinese_characters(self):
        """Test F1 score with Chinese characters"""
        pred = "你好世界"
        ref = "你好中國"
        score = char_f1_score(pred, ref)
        # Common chars: 你, 好 (2 chars)
        # Precision: 2/4 = 0.5
        # Recall: 2/4 = 0.5
        # F1: 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
        self.assertAlmostEqual(score, 0.5, places=5)

    def test_normalized_text_example(self):
        """Test F1 score with real normalized OCR text"""
        pred_original = "檔 號：保存年限"
        ref_original = "檔號：保存年限："

        pred = normalize_text(pred_original)
        ref = normalize_text(ref_original)

        score = char_f1_score(pred, ref)
        # Both should normalize to "檔號保存年限"
        self.assertEqual(pred, "檔號保存年限")
        self.assertEqual(ref, "檔號保存年限")
        self.assertEqual(score, 1.0)

    def test_empty_strings(self):
        """Test F1 score with empty strings"""
        score = char_f1_score("", "")
        self.assertEqual(score, 0.0)

    def test_one_empty_string(self):
        """Test F1 score when one string is empty"""
        score1 = char_f1_score("ABC", "")
        score2 = char_f1_score("", "XYZ")
        self.assertEqual(score1, 0.0)
        self.assertEqual(score2, 0.0)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
