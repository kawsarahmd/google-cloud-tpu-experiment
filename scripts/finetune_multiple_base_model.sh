#!/bin/bash

base_models=("kawsarahmd/bnT5_base_v1_128k_vocab" "kawsarahmd/bnT5_base_v1_64k_vocab" "kawsarahmd/bnT5_base_v1_32k_vocab")
suffix="bangla_summary_v1"
output_dir="~/transformers_models_finetuned"
dataset_name="kawsarahmd/papers_summary_datasets_v3"
user_name="kawsarahmd"
hf_token=""
total_start=$(date +%s)

for base_model in "${base_models[@]}"; do
    start_time=$(date +%s)
    fine_tuned_model_name=$(basename "$base_model")_"$suffix"
    model_output_dir="$output_dir/$fine_tuned_model_name"

    python z-flax-summarization_v3.py \
        --output_dir "$model_output_dir" \
        --model_name_or_path "$base_model" \
        --tokenizer_name "$base_model" \
        --dataset_name="$dataset_name" \
        --do_train --do_eval --predict_with_generate \
        --num_train_epochs 1 \
        --learning_rate 5e-5 --warmup_steps 0 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --overwrite_output_dir \
        --max_source_length 512 --max_target_length 256 \
        --trust_remote_code=True \
        --push_to_hub=False \
        --overwrite_output \
        --hub_token="$hf_token" \
        --hub_model_id="$user_name/$fine_tuned_model_name"

    end_time=$(date +%s)
    duration=$((end_time - start_time))
    hours=$(echo "$duration / 3600" | bc -l)

    if [ $? -eq 0 ]; then
        printf "Fine-tuning completed for $base_model with suffix $suffix in %.2f hours.\n" $hours
    else
        echo "Error: Fine-tuning failed for $base_model with suffix $suffix"
        exit 1
    fi
done

total_end=$(date +%s)
total_duration=$((total_end - total_start))
total_hours=$(echo "$total_duration / 3600" | bc -l)
printf "All fine-tuning processes are completed. Total time: %.2f hours.\n" $total_hours
