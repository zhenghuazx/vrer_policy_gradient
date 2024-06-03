method=trpovrer
task=CartPole-v1
envs=12
buffer_size=400
num_reuse=3
max_steps=1000000

selections=(1.4 1.6)

for c in "${selections[@]}"; do
    for seed in {2021..2025}; do
    	checkpoint="/work/ai-biodigital/hua.zheng/vrer-pg/sensitivity/grad_variance/${task}/${method}/${seed}/c${c}/buffer${buffer_size}"
    	# Check if the directory exists
        if [ -d "$checkpoint" ]; then
            echo "Directory $checkpoint already exists, skipping..."
            continue
        fi
    	vrer-pg train $method --n-env $envs --env $task --lr 0.0003 --mini-batches 128 --n-steps 128 --clip-norm 0.2 --save_grad_variance --selection-constant $c --seed $seed --history-checkpoint $checkpoint --buffer_size $buffer_size --num_reuse_each_iter ${num_reuse} --max-steps ${max_steps}
    done
done
