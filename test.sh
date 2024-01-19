#!/bin/bash

python run2.py \
    --output_dir saved_models \
    --model_name_or_path microsoft/unixcoder-base-nine \
    --do_test \
    --test_data_file ../dataset/test.jsonl \
    --image_path ../dataset/image \
    --block_size 400 \
    --eval_batch_size 64 \
    --seed 99 2>&1