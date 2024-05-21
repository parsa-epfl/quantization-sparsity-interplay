sparsity_num_format=bfp
mantbits=5
sparsify=True
sparsity_mode='unstructured'

rearrange=False
first='q'
bfloat=16
scale_bits=8

sparsity_frac=0.5
N="[2]"
M="[4]"

unconstrained=False
bit_range="[]"

if [ $sparsity_num_format == bfp ]; then
	blocksize=64
else
	blocksize=32
fi

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
   bfloat: $bfloat
   scale_bits: $scale_bits
   device: 'cuda'" >> ../../../src/transformers/bfp/bfp_config.yaml

cd ../../../
pip install -e .
   
cd examples/pytorch/language-modeling
CUDA_VISIBLE_DEVICES=0 python3 run_opt.py \
    --model_name_or_path facebook/opt-125m \
    --tokenizer_name facebook/opt-125m \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --seed 14 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /scratch/kostenok/experiments/opt-125/$sparsity_num_format/$mant_bits/$seed \
    --overwrite_output_dir \
    --learning_rate 1e-04 \
    --adam_beta1 0.9  \
    --adam_beta2 0.999  \
    --adam_epsilon 1e-08  \
    --lr_scheduler_type linear \
    --num_train_epochs 3 \
