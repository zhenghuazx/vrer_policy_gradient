#!/bin/bash
echo "seed: $1"

seed=$1
method=trpovrer
task=HopperBulletEnv-v0
envs=6
buffer_size=400
num_reuse=2
c=1.08
max_steps=2000000

checkpoint="${task}/${method}/new/${seed}/c${c}/envs${envs}/buffer${buffer_size}"
# Check if the directory exists
if [ -d "$checkpoint" ]; then
    echo "Directory $checkpoint already exists, skipping..."
else
    vrer-pg train $method --n-env $envs --env $task --lr 0.0003 --mini-batches 128 --n-steps 128 --clip-norm 0.2 --selection-constant $c --seed $seed --history-checkpoint $checkpoint --buffer_size $buffer_size --num_reuse_each_iter ${num_reuse} --max-steps ${max_steps}
fi
