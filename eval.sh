#!/bin/bash

MODEL_NAME=phi-2
STAGE=sft
EPOCH=1.0 #3.0
DATA=peer_read_train
SAVE_PATH=./models/$STAGE/$MODEL_NAME-$STAGE-$DATA-$EPOCH
SAVE_PATH_PREDICT=$SAVE_PATH/Predict
MODEL_PATH=./models/$MODEL_NAME
LoRA_TARGET=q_proj,v_proj
TEMPLATE=default
PREDICTION_SAMPLES=20
BATCH_SIZE=8

if [ ! -d $MODEL_PATH ]; then
    echo "Model not found: $MODEL_PATH"
    exit 1
fi

if [ ! -d $SAVE_PATH ]; then
    mkdir -p $SAVE_PATH
fi

if [ ! -d $SAVE_PATH_PREDICT ]; then
    mkdir -p $SAVE_PATH_PREDICT
fi

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage $STAGE \
    --model_name_or_path $MODEL_PATH \
    --do_predict \
    --max_samples $PREDICTION_SAMPLES \
    --predict_with_generate \
    --dataset $DATA \
    --template $TEMPLATE \
    --finetuning_type lora \
    --adapter_name_or_path $SAVE_PATH \
    --output_dir $SAVE_PATH_PREDICT \
    --per_device_eval_batch_size 1 \
    |& tee $SAVE_PATH_PREDICT/predict_log.txt