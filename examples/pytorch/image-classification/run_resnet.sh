#!/bin/bash

# This script runs evaluation for MXFP8 and MXFP6 for 2:4 and 50% sparsity on ViT

sparsity_num_format=mx
mantbits=8
sparsify=True
sparsity_mode='unstructured'
mx_w_elem_format='fp8_e4m3'
mx_a_elem_format='fp8_e4m3'
benchmark=imagenet-1k

rearrange=False
first='s'
bfloat=16
scale_bits=8

sparsity_frac=0.5
N="[2]"
M="[4]"

unconstrained=False
bit_range="[2,3]"

if [ $sparsity_num_format == bfp ]; then
	blocksize=64
else
	blocksize=32
fi

if [ $sparsify == True ]; then
	if [ $sparsity_mode == 'unstructured' ]; then
		model="/scratch/new_ayan/checkpoints/resnet50-$benchmark/fp32_unstructured_$sparsity_frac"
		suffix="${sparsity_frac}"
	else
		model="/scratch/new_ayan/checkpoints/resnet50-$benchmark/fp32_structured_$N$M"
		suffix="${N}_${M}"
	fi
else
	model="/scratch/new_ayan/checkpoints/$benchmark/fp32_dense"
	suffix="$epochs"
fi

filename=resnet50\_eval\_$sparsity_num_format\_$mantbits\_$blocksize/$suffix
logfile=resnet50\_eval\_$benchmark\_log\_$sparsity_num_format\_$mantbits\_$blocksize\_$suffix.txt

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

mantbits=6
sparsity_mode='unstructured'
mx_w_elem_format='fp6_e2m3'
mx_a_elem_format='fp6_e2m3'

if [ $sparsity_num_format == bfp ]; then
	blocksize=64
else
	blocksize=32
fi

if [ $sparsify == True ]; then
	if [ $sparsity_mode == 'unstructured' ]; then
		model="/scratch/new_ayan/checkpoints/resnet50-$benchmark/fp32_unstructured_$sparsity_frac"
		suffix="${sparsity_frac}"
	else
		model="/scratch/new_ayan/checkpoints/resnet50-$benchmark/fp32_structured_$N$M"
		suffix="${N}_${M}"
	fi
else
	model="/scratch/new_ayan/checkpoints/$benchmark/fp32_dense"
	suffix="$epochs"
fi

filename=resnet50\_eval\_$sparsity_num_format\_$mantbits\_$blocksize/$suffix
logfile=resnet50\_eval\_$benchmark\_log\_$sparsity_num_format\_$mantbits\_$blocksize\_$suffix.txt

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


mantbits=8
sparsity_mode='structured'
mx_w_elem_format='fp8_e4m3'
mx_a_elem_format='fp8_e4m3'

if [ $sparsity_num_format == bfp ]; then
	blocksize=64
else
	blocksize=32
fi

if [ $sparsify == True ]; then
	if [ $sparsity_mode == 'unstructured' ]; then
		model="/scratch/new_ayan/checkpoints/resnet50-$benchmark/fp32_unstructured_$sparsity_frac"
		suffix="${sparsity_frac}"
	else
		model="/scratch/new_ayan/checkpoints/resnet50-$benchmark/fp32_structured_$N$M"
		suffix="${N}_${M}"
	fi
else
	model="/scratch/new_ayan/checkpoints/$benchmark/fp32_dense"
	suffix="$epochs"
fi

filename=resnet50\_eval\_$sparsity_num_format\_$mantbits\_$blocksize/$suffix
logfile=resnet50\_eval\_$benchmark\_log\_$sparsity_num_format\_$mantbits\_$blocksize\_$suffix.txt

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


mantbits=6
sparsity_mode='structured'
mx_w_elem_format='fp6_e2m3'
mx_a_elem_format='fp6_e2m3'

if [ $sparsity_num_format == bfp ]; then
	blocksize=64
else
	blocksize=32
fi

if [ $sparsify == True ]; then
	if [ $sparsity_mode == 'unstructured' ]; then
		model="/scratch/new_ayan/checkpoints/resnet50-$benchmark/fp32_unstructured_$sparsity_frac"
		suffix="${sparsity_frac}"
	else
		model="/scratch/new_ayan/checkpoints/resnet50-$benchmark/fp32_structured_$N$M"
		suffix="${N}_${M}"
	fi
else
	model="/scratch/new_ayan/checkpoints/$benchmark/fp32_dense"
	suffix="$epochs"
fi

filename=resnet50\_eval\_$sparsity_num_format\_$mantbits\_$blocksize/$suffix
logfile=resnet50\_eval\_$benchmark\_log\_$sparsity_num_format\_$mantbits\_$blocksize\_$suffix.txt

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
