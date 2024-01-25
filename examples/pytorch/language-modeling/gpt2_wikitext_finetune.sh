blocksize=64
mantbits=7
sparsity_frac=0.6
num_format=bfp
sparsity_num_format=bfp
rearrange=False
N="[-2]"
M="[-2]"

rm /usr/local/lib/python3.8/dist-packages/transformers/bfp/bfp_config.yaml
rm /usr/local/lib/python3.8/dist-packages/transformers/bfp/bfp_ops.py
cp ../../../src/transformers/bfp/bfp_ops.py /usr/local/lib/python3.8/dist-packages/transformers/bfp/
cp ../../../src/transformers/models/gpt2/modeling_gpt2.py /usr/local/lib/python3.8/dist-packages/transformers/models/gpt2/
echo -e "hbfp:
   num_format: '$num_format'
   sparsity_num_format: '$sparsity_num_format' 
   rounding_mode: 'stoc' 
   epsilon: 0.00000001 
   mant_bits: $mantbits 
   weight_mant_bits: 15 
   bfp_tile_size: 8 
   bfp_block_size: $blocksize 
   in_sparsity: False
   w_sparsity: False 
   grad_sparsity: False
   rearrange: $rearrange
   sparsity_frac: $sparsity_frac
   N: $N
   M: $M 
   unconstrained: $unconstrained
   bit_range: $bit_range
   device: 'cuda'" >> /usr/local/lib/python3.8/dist-packages/transformers/bfp/bfp_config.yaml

cd ../../../
cd examples/pytorch/language-modeling

python3 run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_eval \
    --output_dir ./tmp/test-clm \
    --overwrite_output_dir \
    --num_train_epochs 0.1 \
    --do_train \
