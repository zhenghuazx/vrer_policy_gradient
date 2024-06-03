method=a2cvrer
task=CartPole-v1
envs=24
buffer_size=(800 1000 1500)
num_reuse=3
max_steps=1000000
c=1.05


for bs in "${buffer_size[@]}"; do
    for seed in {2021..2025}; do
    	checkpoint="/work/ai-biodigital/hua.zheng/vrer-pg/sensitivity/buffer/${task}/${method}/${seed}/c${c}/buffer${bs}"
    	# Check if the directory exists
        if [ -d "$checkpoint" ]; then
            echo "Directory $checkpoint already exists, skipping..."
            continue
        fi
    	vrer-pg train $method --n-env $envs --env $task --lr 0.0003 --n-steps 16 --clip-norm 0.2 --selection-constant $c --seed $seed --history-checkpoint $checkpoint --buffer_size $bs --num_reuse_each_iter ${num_reuse} --max-steps ${max_steps}
    done
done
