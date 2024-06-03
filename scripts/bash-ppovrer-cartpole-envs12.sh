#!/bin/bash
echo "seed: $1"

seed=$1

method=ppovrer
task=CartPole-v1
envs=12
buffer_size=400
num_reuse=3
c=1.05
max_steps=2000000

checkpoint="/work/ai-biodigital/hua.zheng/vrer-pg/${task}/${method}/${seed}/envs${envs}/buffer${buffer_size}"
# Check if the directory exists
if [ -d "$checkpoint" ]; then
    echo "Directory $checkpoint already exists, skipping..."
    continue
fi
vrer-pg train $method --n-env $envs --env $task --lr 0.0003 --mini-batches 128 --n-steps 128 --clip-norm 0.2 --selection-constant $c --seed $seed --history-checkpoint $checkpoint --buffer_size $buffer_size --num_reuse_each_iter ${num_reuse} --max-steps ${max_steps}
