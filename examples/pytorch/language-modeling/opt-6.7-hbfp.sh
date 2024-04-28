compute_node=$1
blocksize=64
mantbits=7
sparsity_frac=0.5
num_format=bfp
sparsity_num_format=fp32
rearrange=False

N="[2]"
M="[4]"

unconstrained=False
bit_range="[]"

   if [ $sparsity_num_format == "fp32" ]
   then
      filename=$sparsity_num_format/fp32\_$N:$M
   else
      filename=$sparsity_num_format\_block\_size\_$blocksize/hbfp\_$bit_range/$benchmark\_bfp$mantbits\_sparse\_$blocksize
      mkdir /parsadata1/lisa/experiments/magn_based/6.7b/fp_2:4/val/
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
   w_sparsity: True
   grad_sparsity: False
   rearrange: $rearrange
   sparsity_frac: $sparsity_frac
   N: $N
   M: $M
   unconstrained: $unconstrained
   bit_range: $bit_range
   device: 'cuda'" >> ../../../src/transformers/bfp/bfp_config.yaml
      cd ../../../
      pip install -e .
   fi
cd examples/pytorch/language-modeling
python3 run_opt.py \
    --model_name_or_path /parsadata1/lisa/experiments/magn_based/6.7b/fp_2:4/checkpoint-1000/ \
    --tokenizer_name facebook/opt-6.7b \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --do_eval \
    --remove_unused_columns False \
    --output_dir /parsadata1/lisa/experiments/magn_based/6.7b/fp_2:4/val/ \
    --overwrite_output_dir \
    --learning_rate 5e-04 \
    --adam_beta1 0.9  \
    --adam_beta2 0.999  \
    --adam_epsilon 1e-08  \
    --lr_scheduler_type linear \
    --num_train_epochs 1 \
    --max_steps 1000 \
