#!/bin/bash

sparsity_num_format=fp32
mantbits=7

sparsify=False
first='s'
sparsity_mode='structured'
mx_w_elem_format='fp8_e4m3'
mx_a_elem_format='fp8_e4m3'

sparsity_frac=0.5
N=2
M=4
epochs=3

benchmark=imagenet-1k
model='google/vit-base-patch16-224'
optim=adamw_hf

if [ $sparsity_num_format == bfp ]; then
	blocksize=64
else
	blocksize=32
fi

filename=vit\_eval\_chkpt
logfile=vit\_eval\_log.txt

rm ../../../src/transformers/bfp/bfp_config.yaml
echo -e "hbfp:
   num_format: 'bfp'
   sparsity_num_format: '$sparsity_num_format' 
   rounding_mode: 'stoc' 
   epsilon: 0.00000001 
   mant_bits: $mantbits 
   weight_mant_bits: 15 
   block_size: $blocksize 
   in_sparsity: False
   w_sparsity: $sparsify 
   grad_sparsity: False
   sparsity_frac: $sparsity_frac
   N: $N
   M: $M
   first: $first
   sparsity_mode: $sparsity_mode
   mx_w_elem_format: $mx_w_elem_format
   mx_a_elem_format: $mx_a_elem_format
   bfloat: 16
   scale_bits: 8
   device: 'cuda'" >> ../../../src/transformers/bfp/bfp_config.yaml

cd ../../../
pip install -e .

cd examples/pytorch/image-classification/
python3 run_image_classification.py  \
   --model_name_or_path $model \
   --dataset_name $benchmark  \
   --output_dir ./$benchmark\_results/$filename  \
   --use_auth_token=True \
   --overwrite_output_dir \
   --remove_unused_columns False  \
   --num_train_epochs $epochs \
   --do_eval  \
   --learning_rate 5e-5  \
   --per_device_train_batch_size 8  \
   --per_device_eval_batch_size 16  \
   --logging_strategy steps  \
   --logging_steps 10  \
   --evaluation_strategy epoch  \
   --save_strategy epoch  \
   --load_best_model_at_end True  \
   --save_total_limit 5  \
   --seed 42  \
   --gradient_accumulation_steps 4  \
   --adam_beta1 0.9  \
   --adam_beta2 0.999  \
   --adam_epsilon 1e-08  \
   --lr_scheduler_type linear \
   --optim $optim | tee $logfile
#--do_train
#--num_train_epochs $epochs
