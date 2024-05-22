sparsity_num_format=bfp
mantbits=7

sparsify=True
first='q'
sparsity_mode='structured'
mx_w_elem_format='fp8_e4m3'
mx_a_elem_format='fp8_e4m3'

sparsity_frac=0.5
N=2
M=4
epochs=3

model='meta-llama/Llama-2-70b-hf'

if [ $sparsity_num_format == bfp ]; then
	blocksize=64
else
	blocksize=32
fi

filename=llama2\_fn_hbfp8\_chkpt

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
   sparsity_frac: $sparsity_frac
   N: $N
   M: $M
   unconstrained: $unconstrained
   first: $first
   sparsity_mode: $sparsity_mode
   bit_range: $bit_range
   bfloat: 16
   scale_bits: 8
   device: 'cuda'" >> ../../../src/transformers/bfp/bfp_config.yaml

cd ../../../
pip install -e .
cd examples/pytorch/language-modeling
python3 run_llama.py \
    --model_name_or_path $model  \
    --tokenizer_name $model \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --max_steps 60 \
    --save_steps 60 \
    --logging_steps 10 \
    --remove_unused_columns True \
    --output_dir $filename \
    --overwrite_output_dir \
    --learning_rate 5e-05 \
    --adam_beta1 0.9  \
    --adam_beta2 0.999  \
    --adam_epsilon 1e-08  \
    --gradient_accumulation_steps 1 \
    --max_grad_norm 0.3 \
    --weight_decay 0.001 \
    --lr_scheduler_type linear \
    --num_train_epochs 3 \
    --optim paged_adamw_32bit \
