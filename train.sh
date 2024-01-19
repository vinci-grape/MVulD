#!/bin/bash

python run.py \
--output_dir saved_models \
--model_name_or_path microsoft/unixcoder-base-nine \
--do_train \
--train_data_file ../dataset/train.jsonl \
--eval_data_file ../dataset/valid.jsonl \
--image_path ../dataset/image \
--num_train_epochs 200 \
--block_size 400 \
--train_batch_size 64 \
--eval_batch_size 64 \
--learning_rate 5e-5 \
--max_grad_norm 1.0 \
--weight_decay 0.005 \
--seed 99 2>&1