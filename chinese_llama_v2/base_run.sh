accelerate launch python train.py \
    --model_name_or_path /root/autodl-tmp/llama-chinese-7b \
    --data_path data_proj/opendata \
    --bf16 False \
    --output_dir /root/autodl-fs/output_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 20 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --logging_steps 10 \
    --tf32 False \
    --model_max_length 2048