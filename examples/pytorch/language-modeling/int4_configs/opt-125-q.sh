sparsity_num_format=int
mantbits=4

sparsify=True
first='s'
sparsity_mode='structured'
mx_w_elem_format='fp8_e4m3'
mx_a_elem_format='fp8_e4m3'

sparsity_frac=0.5
N=2
M=4
epochs=3

model='facebook/opt-125m'

if [ $sparsity_num_format == bfp ]; then
	blocksize=64
else
	blocksize=1
fi

filename=opt-125\ff1_int4\_fp16_quant\_block=$blocksize

rm ../../../../src/transformers/bfp/bfp_config.yaml
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
   unconstrained: $unconstrained
   first: $first
   sparsity_mode: $sparsity_mode
   bit_range: $bit_range
   bfloat: 16
   scale_bits: 8
   device: 'cuda'" >> ../../../../src/transformers/bfp/bfp_config.yaml

cd ../../../../
pip install -e .
   
cd examples/pytorch/language-modeling
CUDA_VISIBLE_DEVICES=0 python3 run_opt.py \
    --model_name_or_path /scratch/kostenok/transformers_hbfp_sparsity/examples/pytorch/language-modeling/opt-125_ff1_2:4_ff2_1:4_11_2:4_no_warmup \
    --tokenizer_name $model \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --seed 14 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_eval \
    --output_dir $filename \
    --overwrite_output_dir \
    --learning_rate 1e-04 \
    --adam_beta1 0.9  \
    --adam_beta2 0.999  \
    --adam_epsilon 1e-08  \
    --lr_scheduler_type linear \
    --num_train_epochs 2.7 \
    --optim paged_adamw_32bit \
