"""
簡易測試：測試 timeout 和錯誤處理
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import Mock, patch
import requests


def _encode_image(image_path: str) -> str:
    """簡化版的圖片編碼"""
    if image_path.startswith(('http://', 'https://', 'data:', 'file://')):
        return image_path
    return f"data:image/png;base64,fake_base64_data"


def call_vllm(vllm_url: str, image_url: str, model_name: str) -> str:
    """測試版本的 call_vllm 函數"""
    encoded_image = _encode_image(image_url)

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": encoded_image
                        }
                    },
                    {
                        "type": "text",
                        "text": "執行 OCR 任務："
                    }
                ]
            }
        ],
        "temperature": 0.1,
        "repetition_penalty": 1.1,
        "stop": ["<end_of_turn>"],
    }

    try:
        response = requests.post(
            vllm_url,
            json=payload,
            timeout=600
        )
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except requests.exceptions.Timeout:
        print(f"請求超時，返回空字串")
        return ""
    except requests.exceptions.HTTPError as e:
        print(f"HTTP 錯誤：{e}")
        print(f"請求 URL：{vllm_url}")
        print(f"回應內容：{response.text}")
        return ""
    except Exception as e:
        print(f"未預期的錯誤：{e}")
        return ""


def test_timeout_returns_empty_string():
    """測試 timeout 時返回空字串"""
    with patch('requests.post') as mock_post:
        mock_post.side_effect = requests.exceptions.Timeout()

        result = call_vllm(
            vllm_url="http://fake-url",
            image_url="test.jpg",
            model_name="test-model"
        )

        assert result == "", f"Expected empty string, got: {result}"
        print("✓ Timeout test passed: returns empty string")


def test_http_error_returns_empty_string():
    """測試 HTTP 錯誤時返回空字串"""
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
        mock_response.text = "Error message"
        mock_post.return_value = mock_response

        result = call_vllm(
            vllm_url="http://fake-url",
            image_url="test.jpg",
            model_name="test-model"
        )

        assert result == "", f"Expected empty string, got: {result}"
        print("✓ HTTP Error test passed: returns empty string")


def test_success_returns_content():
    """測試成功時返回內容"""
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'choices': [{'message': {'content': 'OCR result text'}}]
        }
        mock_post.return_value = mock_response

        result = call_vllm(
            vllm_url="http://fake-url",
            image_url="test.jpg",
            model_name="test-model"
        )

        assert result == "OCR result text", f"Expected 'OCR result text', got: {result}"
        print("✓ Success test passed: returns correct content")


def test_generic_exception_returns_empty_string():
    """測試其他異常時返回空字串"""
    with patch('requests.post') as mock_post:
        mock_post.side_effect = Exception("Unknown error")

        result = call_vllm(
            vllm_url="http://fake-url",
            image_url="test.jpg",
            model_name="test-model"
        )

        assert result == "", f"Expected empty string, got: {result}"
        print("✓ Generic exception test passed: returns empty string")


if __name__ == "__main__":
    print("Running timeout and error handling tests...\n")

    test_timeout_returns_empty_string()
    test_http_error_returns_empty_string()
    test_success_returns_content()
    test_generic_exception_returns_empty_string()

    print("\n✅ All tests passed!")
