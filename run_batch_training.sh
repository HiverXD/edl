#!/bin/bash

# SPECTRA Batch Training Automator
# Iterates through all mazes and multiple seeds.

# Usage: ./run_batch_training.sh [reward_type]
REWARD_TYPE=${1:-"static"}

# Base mazes found in logs/laplacian_encoder/
ALL_MAZES=$(ls -d logs/laplacian_encoder/*/ | xargs -n 1 basename)

# Seeds to run for each maze
SEEDS=(42 123 999)

# Mazes to exclude
EXCLUDE=("nothing")

echo "--- Starting SPECTRA Batch Training ---"
echo "Reward Type: $REWARD_TYPE"
echo "Seeds:       ${SEEDS[*]}"

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
        echo "Skipping excluded maze: $MAZE"
        continue
    fi

    echo ""
    echo "============================================"
    echo "  MAZE: $MAZE"
    echo "============================================"
    
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo ">>> [Seed $SEED] Training $MAZE with $REWARD_TYPE..."
        
        # 1. Create a temporary config for this maze
        TMP_CONFIG="configs/tmp_${MAZE}_${SEED}.yaml"
        cp config.yaml $TMP_CONFIG
        
        # Update maze_type in the temp config
        sed -i "s/maze_type: \".*\"/maze_type: \"$MAZE\"/" $TMP_CONFIG
        
        # 2. Run Training
        python train_gasd_sac.py --config $TMP_CONFIG --reward_type "$REWARD_TYPE" --seed "$SEED"
        
        if [ $? -eq 0 ]; then
            echo ">>> [Success] Seed $SEED for $MAZE finished."
        else
            echo ">>> [Error] Seed $SEED for $MAZE failed."
        fi
        
        # 3. Cleanup temp config
        rm $TMP_CONFIG
    done
done

echo ""
echo "--- All Batch Training Completed! ---"
