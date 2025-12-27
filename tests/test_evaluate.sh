#!/bin/bash

# Evaluation script for Gemma-3-12b models
# Compare performance before and after SFT

TEST_DATA="data/input/ocr_non_test_data=2700.json"
VLLM_URL="http://192.168.1.78:3472/v1/chat/completions"
MODEL_NAME="gemma-12b"
WORKERS=2
TOP_K=1  # Use all test data

echo "=========================================="
echo "Gemma-3-12b Model Evaluation"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Test Data: ${TEST_DATA}"
echo "  Test Set Size: ${TOP_K}"
echo "  VLLM URL: ${VLLM_URL}"
echo "  Model Name: ${MODEL_NAME}"
echo "  Workers: ${WORKERS}"
echo ""

# Deploy SFT model for Evaluation 1
echo "=========================================="
echo "Deploying SFT model..."
echo "=========================================="
echo ""
echo "1. Stopping baseline model (gemma-12b-full)..."
docker compose -f /tmp2/howard/model-deploy-tool/vllm/docker-compose-gemma-12b-full.yml down

echo ""
echo "2. Starting SFT model..."
docker compose -f /tmp2/howard/model-deploy-tool/vllm/gemma3-12b-sft/docker-compose-gemma12b-merge.yml up -d

echo ""
echo "3. Waiting 120 seconds for model to load..."
for i in {120..1}; do
    printf "\r   Time remaining: %3d seconds" $i
    sleep 1
done
echo ""
echo ""
echo "✓ SFT model deployed!"
echo ""

# Evaluation 1: Gemma3 after SFT (lr=2e-4, e=5)
echo "=========================================="
echo "Evaluation 1: Gemma3 after SFT (lr=2e-4, e=5)"
echo "=========================================="
echo ""

python core/evaluate.py \
  --vllm-url ${VLLM_URL} \
  --data-file ${TEST_DATA} \
  --top-k ${TOP_K} \
  --model-name ${MODEL_NAME} \
  --output-file results/gemma3_sft_lr2e4_ep5_test2700.json \
  --workers ${WORKERS} \
  --save-interval 100

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Evaluation 1 completed successfully!"
    echo "  Results saved to: results/gemma3_sft_lr2e4_ep5_test2700.json"
else
    echo ""
    echo "✗ Evaluation 1 failed!"
    exit 1
fi

echo ""
echo ""

# Switch to baseline model
echo "=========================================="
echo "Switching to baseline model..."
echo "=========================================="
echo ""
echo "1. Stopping SFT model..."
docker compose -f /tmp2/howard/model-deploy-tool/vllm/gemma3-12b-sft/docker-compose-gemma12b-merge.yml down

echo ""
echo "2. Starting baseline model (gemma-12b-full)..."
docker compose -f /tmp2/howard/model-deploy-tool/vllm/docker-compose-gemma-12b-full.yml up -d

echo ""
echo "3. Waiting 120 seconds for model to load..."
for i in {120..1}; do
    printf "\r   Time remaining: %3d seconds" $i
    sleep 1
done
echo ""
echo ""
echo "✓ Model switch completed!"
echo ""

# Evaluation 2: Gemma3 before SFT
echo "=========================================="
echo "Evaluation 2: Gemma3 before SFT (baseline)"
echo "=========================================="
echo ""

python core/evaluate.py \
  --vllm-url ${VLLM_URL} \
  --data-file ${TEST_DATA} \
  --top-k ${TOP_K} \
  --model-name ${MODEL_NAME} \
  --output-file results/gemma3_before_sft_test2700.json \
  --workers ${WORKERS} \
  --save-interval 100

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Evaluation 2 completed successfully!"
    echo "  Results saved to: results/gemma3_before_sft_test2700.json"
else
    echo ""
    echo "✗ Evaluation 2 failed!"
    exit 1
fi

echo ""
echo ""

# Summary
echo "=========================================="
echo "✓ All Evaluations Completed!"
echo "=========================================="
echo ""
echo "Results:"
echo "  1. After SFT (lr=2e-4, e=5):  results/gemma3_sft_lr2e4_ep5_test2700.json"
echo "  2. Before SFT (baseline):     results/gemma3_before_sft_test2700.json"
echo ""
echo "Next steps:"
echo "  - Compare results: python ui/compare_finetune.py"
echo ""
