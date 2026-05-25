#!/bin/bash
# scripts/run_loocv.sh
# Runs full Leave-One-Out Cross Validation across all 23 RESECT patients.
# Each fold: 1 patient = test, 1 = val, 21 = train.
# Usage: bash scripts/run_loocv.sh

set -e

DATA_DIR="data/processed"
CONFIG="configs/train_config.yaml"
N_CASES=23

echo "=========================================="
echo " Brain Shift Prediction — LOOCV Training"
echo " Patients: $N_CASES"
echo "=========================================="

mkdir -p data/splits
mkdir -p outputs/loocv
mkdir -p eval_results/loocv

TRE_RESULTS=()

for FOLD in $(seq 0 $((N_CASES - 1))); do
    echo ""
    echo "─── FOLD $FOLD / $((N_CASES - 1)) ───────────────────"

    SPLIT_FILE="data/splits/loocv_fold${FOLD}.csv"
    OUT_DIR="outputs/loocv/fold${FOLD}"
    LOG_DIR="logs/loocv/fold${FOLD}"

    # 1. Create split
    python data/dataset.py --loocv-fold $FOLD \
        --data_dir "$DATA_DIR" \
        --output "$SPLIT_FILE"

    # 2. Train
    python training/train.py \
        --config "$CONFIG" \
        --data_dir "$DATA_DIR" \
        --split_file "$SPLIT_FILE" \
        --output_dir "$OUT_DIR" \
        --log_dir "$LOG_DIR" \
        --epochs 200

    # 3. Evaluate
    python evaluation/metrics.py \
        --checkpoint "$OUT_DIR/best_model.pth" \
        --data_dir "$DATA_DIR" \
        --split test \
        --output_dir "eval_results/loocv/fold${FOLD}"

    echo "Fold $FOLD complete."
done

# Aggregate results
echo ""
echo "=========================================="
echo " Aggregating LOOCV results..."
python scripts/aggregate_loocv.py \
    --eval_dir eval_results/loocv \
    --output   eval_results/loocv_summary.json

echo "Done! Summary: eval_results/loocv_summary.json"
