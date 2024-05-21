compute_node=$1
blocksize=64
mantbits=7
sparsity_frac=0.5
num_format=bfp
sparsity_num_format=bfp
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
      mkdir /scratch/kostenok/experiments/llama3-finetune/
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
   w_sparsity: False
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
CUDA_VISIBLE_DEVICES=0 python3 llama3-one-shot-eval.py \
    --model_name_or_path /scratch/kostenok/llama3-hf-checkpoint \
    --tokenizer_name /scratch/kostenok/llama3-hf-checkpoint \
    --dataset_name google/boolq \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_eval \
    --max_steps 60 \
    --save_steps 60 \
    --logging_steps 10 \
    --remove_unused_columns True \
    --output_dir /scratch/kostenok/experiments/llama3-finetune/ \
    --overwrite_output_dir \
    --learning_rate 5e-05 \
    --adam_beta1 0.9  \
    --adam_beta2 0.999  \
    --adam_epsilon 1e-08  \
    --gradient_accumulation_steps 1 \
    --max_grad_norm 0.3 \
    --weight_decay 0.001 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type linear \
    --num_train_epochs 3 \
    --optim paged_adamw_32bit \
