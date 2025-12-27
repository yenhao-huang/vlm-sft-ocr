"""
最簡易測試：直接驗證 timeout 行為
運行方式: python tests/test_timeout_simple.py
"""

from unittest.mock import Mock, patch
import requests


def call_vllm_with_timeout_handling(vllm_url, payload):
    """帶有 timeout 處理的請求函數"""
    try:
        response = requests.post(vllm_url, json=payload, timeout=600)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except requests.exceptions.Timeout:
        print("請求超時，返回空字串")
        return ""
    except requests.exceptions.HTTPError as e:
        print(f"HTTP 錯誤：{e}")
        return ""
    except Exception as e:
        print(f"未預期的錯誤：{e}")
        return ""


# 測試 1: Timeout
print("測試 1: Timeout 情況")
with patch('requests.post') as mock_post:
    mock_post.side_effect = requests.exceptions.Timeout()
    result = call_vllm_with_timeout_handling("http://test", {})
    assert result == "", "Timeout 應返回空字串"
    print("✓ PASS\n")

# 測試 2: HTTP Error
print("測試 2: HTTP Error 情況")
with patch('requests.post') as mock_post:
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
    mock_response.text = "Error"
    mock_post.return_value = mock_response
    result = call_vllm_with_timeout_handling("http://test", {})
    assert result == "", "HTTP Error 應返回空字串"
    print("✓ PASS\n")

# 測試 3: Success
print("測試 3: 成功情況")
with patch('requests.post') as mock_post:
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {'choices': [{'message': {'content': 'OK'}}]}
    mock_post.return_value = mock_response
    result = call_vllm_with_timeout_handling("http://test", {})
    assert result == "OK", "成功應返回內容"
    print("✓ PASS\n")

print("✅ 所有測試通過！")
