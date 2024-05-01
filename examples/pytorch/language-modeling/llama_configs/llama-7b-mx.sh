compute_node=$1
blocksize=32 # IMPORTANT: set block size here for mx formats
mantbits=5 # Not used for mx
sparsity_frac=0.5 # Not used for mx
num_format=bfp # Not used for mx
sparsity_num_format=bfp # Not used for mx
rearrange=False # Not used for mx

N="[2]"
M="[4]"

unconstrained=False
bit_range="[]"

   if [ $sparsity_num_format == "fp32" ]
   then
      filename=$sparsity_num_format/fp32\_$N:$M
   else
      filename=$sparsity_num_format\_block\_size\_$blocksize/hbfp\_$bit_range/$benchmark\_bfp$mantbits\_sparse\_$blocksize
      mkdir OUTPUT_CHECKPOINT_FOLDER
   fi

   if [ $compute_node == "runai" ]
   then
      rm /usr/local/lib/python3.8/dist-packages/transformers/bfp/bfp_config.yaml
      rm /usr/local/lib/python3.8/dist-packages/transformers/bfp/bfp_ops.py
      cp ../../../src/transformers/bfp/bfp_ops.py /usr/local/lib/python3.8/dist-packages/transformers/bfp/
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
   else
      rm ../../../src/transformers/bfp/bfp_config.yaml
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
   w_sparsity: False           # IMPORTANT! If initialization checkpoint is sparsified, we don't need to sparsify twice
   grad_sparsity: False
   rearrange: $rearrange
   sparsity_frac: $sparsity_frac
   N: $N
   M: $M
   unconstrained: $unconstrained
   bit_range: $bit_range
   sparsity_mode: 'structured'  # Don't need to change, it doesn't change the sparsity pattern if checkpoint is already sparsified
   mx_w_elem_format: 'fp8_e4m3' # weights MXFP8 quantization MXFP8, check another options in transformers_hbfp_sparsity/src/transformers/bfp/formats.py
   mx_a_elem_format: 'fp8_e4m3' # activations MXFP8 quantization, check another options in transformers_hbfp_sparsity/src/transformers/bfp/formats.py
   bfloat: 16
   scale_bits: 8
   device: 'cuda'" >> ../../../src/transformers/bfp/bfp_config.yaml
      cd ../../../
      pip install -e .
   fi
cd examples/pytorch/language-modeling
python3 run_llama.py \
    --model_name_or_path PATH_TO_SPARSIFIED_CHECKPOINT_FOLDER \ 
    --tokenizer_name PATH_TO_SPARSIFIED_CHECKPOINT_FOLDER \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_eval_batch_size 1 \
    --do_eval \
    --block_size 1024 \
    --remove_unused_columns True \
    --output_dir OUTPUT_CHECKPOINT_FOLDER \
