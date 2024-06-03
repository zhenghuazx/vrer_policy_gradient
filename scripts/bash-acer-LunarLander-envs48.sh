#!/bin/bash
echo "seed: $1"

seed=$1
method=acer
task=LunarLander-v2
envs=48
buffer_size=5000
max_steps=2000000

checkpoint="/work/ai-biodigital/hua.zheng/vrer-pg/${task}/${method}/${seed}/envs${envs}/buffer${buffer_size}"
# Check if the directory exists
if [ -d "$checkpoint" ]; then
    echo "Directory $checkpoint already exists, skipping..."
else
    vrer-pg train $method --n-env $envs --env $task --lr 0.0003 --n-steps 16 --clip-norm 0.2 --seed $seed --history-checkpoint $checkpoint --buffer-max-size ${buffer_size} --buffer-batch-size 128 --max-steps ${max_steps}
fi

