#!/bin/bash

# SPECTRA Batch Training Automator
# Iterates through all mazes in logs/laplacian_encoder/ except those already trained.

# Base mazes found in logs/laplacian_encoder/
ALL_MAZES=$(ls -d logs/laplacian_encoder/*/ | xargs -n 1 basename)

# Mazes to exclude
EXCLUDE=("square_corridor" "square_a")

echo "--- Starting SPECTRA Batch Training ---"

for MAZE in $ALL_MAZES; do
    # Check if maze should be excluded
    SKIP=0
    for EX in "${EXCLUDE[@]}"; do
        if [ "$MAZE" == "$EX" ]; then
            SKIP=1
            break
        fi
    done
    
    if [ $SKIP -eq 1 ]; then
        echo "Skipping already trained maze: $MAZE"
        continue
    fi

    echo ""
    echo "============================================"
    echo "  TRAINING MAZE: $MAZE"
    echo "============================================"
    
    # 1. Create a temporary config for this maze
    # We use current config.yaml as a template
    TMP_CONFIG="configs/tmp_${MAZE}.yaml"
    cp config.yaml $TMP_CONFIG
    
    # Use sed to update maze_type in the temp config
    # Matches maze_type: "any_value" and replaces with maze_type: "MAZE"
    sed -i "s/maze_type: ".*"/maze_type: "$MAZE"/" $TMP_CONFIG
    
    echo "[Config] Created $TMP_CONFIG with maze_type: $MAZE"
    
    # 2. Run Training
    python train_gasd_sac.py --config $TMP_CONFIG --reward_type static
    
    if [ $? -eq 0 ]; then
        echo "[Success] Finished training for $MAZE"
    else
        echo "[Error] Training failed for $MAZE. Moving to next..."
    fi
    
    # 3. Cleanup temp config
    rm $TMP_CONFIG
done

echo ""
echo "--- All Batch Training Completed! ---"
