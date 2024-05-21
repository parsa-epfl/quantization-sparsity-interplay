sparsity_num_format=int
mantbits=8
sparsify=True
sparsity_mode='unstructured'

rearrange=False
first='s'
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
CUDA_VISIBLE_DEVICES=0 python3 run_llama.py \
    --model_name_or_path PATH_TO_CHECKPOINT \
    --tokenizer_name /scratch/kostenok/llama3-hf-checkpoint \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_eval \
    --remove_unused_columns True \
    --output_dir PATH_TO_OUTPUT_DIR \
