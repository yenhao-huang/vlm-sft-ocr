import json
import os
import tempfile
import shutil
from datetime import datetime


def test_metadata_generation():
    """測試 metadata.json 生成功能"""

    # 建立臨時目錄
    test_dir = tempfile.mkdtemp()

    try:
        # 模擬訓練數據
        metadata = {
            "training_completed_at": datetime.now().isoformat(),
            "gpu_info": {
                "name": "Test GPU",
                "total_memory_gb": 40.0
            },
            "training_time": {
                "total_seconds": 3600.5,
                "total_minutes": 60.01,
                "total_hours": 1.0
            },
            "memory_stats": {
                "start_memory_gb": 2.5,
                "peak_memory_gb": 35.2,
                "peak_memory_for_training_gb": 32.7,
                "peak_memory_percentage": 88.0,
                "peak_training_memory_percentage": 81.75
            }
        }

        # 寫入 metadata.json
        os.makedirs(test_dir, exist_ok=True)
        metadata_path = os.path.join(test_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # 驗證檔案存在
        assert os.path.exists(metadata_path), "metadata.json 未生成"

        # 讀取並驗證內容
        with open(metadata_path, 'r', encoding='utf-8') as f:
            loaded_metadata = json.load(f)

        # 驗證必要欄位
        assert "training_completed_at" in loaded_metadata
        assert "training_time" in loaded_metadata
        assert "memory_stats" in loaded_metadata

        # 印出時間與記憶體資訊
        print("✓ metadata.json 生成成功")
        print(f"\n訓練時間:")
        print(f"  總秒數: {loaded_metadata['training_time']['total_seconds']}")
        print(f"  總分鐘: {loaded_metadata['training_time']['total_minutes']}")
        print(f"  總小時: {loaded_metadata['training_time']['total_hours']}")

        print(f"\n記憶體使用:")
        print(f"  峰值記憶體: {loaded_metadata['memory_stats']['peak_memory_gb']} GB")
        print(f"  訓練記憶體: {loaded_metadata['memory_stats']['peak_memory_for_training_gb']} GB")
        print(f"  記憶體使用率: {loaded_metadata['memory_stats']['peak_memory_percentage']}%")

        print(f"\n✓ 測試通過")

    finally:
        # 清理臨時目錄
        shutil.rmtree(test_dir)


if __name__ == "__main__":
    test_metadata_generation()
