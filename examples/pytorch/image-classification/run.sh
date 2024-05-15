#!/bin/bash

blocksize=32
mantbits=8
sparsity_frac=0.5
sparsity_num_format=int
benchmark=imagenet-1k
rearrange=False
sparsify=True
model='/scratch/new_ayan/transformers_hbfp_sparsity/examples/pytorch/image-classification/imagenet-1k_results/fp32_5_64/0.5_2'
epochs=3
first='s'
sparsity_mode='unstructured'
mx_w_elem_format='fp8_e4m3'
mx_a_elem_format='fp8_e4m3'
bfloat=16
scale_bits=8

N="[2]"
M="[4]"

unconstrained=False
bit_range="[2,3]"

# filename=$sparsity_num_format\_$mantbits\_$blocksize/$N\_$M\_$epochs
# logfile=$benchmark\_log\_$sparsity_num_format\_$mantbits\_$blocksize\_$N\_$M\_$epochs.txt

filename=eval_$sparsity_num_format\_$mantbits\_$blocksize/$sparsity_frac\_$epochs
logfile=eval_$benchmark\_log\_$sparsity_num_format\_$mantbits\_$blocksize\_$sparsity_frac\_$epochs.txt

rm ../../../src/transformers/bfp/bfp_config.yaml
echo -e "hbfp:
   num_format: 'bfp'
   sparsity_num_format: '$sparsity_num_format' 
   rounding_mode: 'stoc' 
   epsilon: 0.00000001 
   mant_bits: $mantbits 
   weight_mant_bits: 15 
   bfp_tile_size: 8 
   bfp_block_size: $blocksize 
   in_sparsity: False
   w_sparsity: $sparsify 
   grad_sparsity: False
   rearrange: $rearrange
   sparsity_frac: $sparsity_frac
   N: $N
   M: $M
   unconstrained: $unconstrained
   first: $first
   sparsity_mode: $sparsity_mode
   bit_range: $bit_range
   mx_w_elem_format: $mx_w_elem_format
   mx_a_elem_format: $mx_a_elem_format
   bfloat: $bfloat
   scale_bits: $scale_bits
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
   --do_eval  \
   --learning_rate 5e-5  \
   --num_train_epochs $epochs  \
   --per_device_train_batch_size 8  \
   --per_device_eval_batch_size 8  \
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
   --optim BFPAdam | tee $logfile
   # --do_train
