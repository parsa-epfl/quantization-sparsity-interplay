sparsity_num_format=bfp
mantbits=5

sparsify=True
first='s'
sparsity_mode='unstructured'
mx_w_elem_format='fp8_e4m3'
mx_a_elem_format='fp8_e4m3'

sparsity_frac=0.5
N=2
M=4
epochs=3

model='llama3-sparse-fn-unstr'
filename=llama3\_eval_hbfp6_unstr\_chkpt

if [ $sparsity_num_format == bfp ]; then
	blocksize=64
else
	blocksize=32
fi

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
CUDA_VISIBLE_DEVICES=0 python3 run_llama.py \
    --model_name_or_path $model \
    --tokenizer_name $model \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_eval \
    --remove_unused_columns True \
    --output_dir $filename \
