if [ $# -lt 1 ]; then
  echo 1>&2 "$0: not enough arguments"
  exit 2
fi
compute_node=$1
blocksize=64
mantbits=7
sparsity_frac=0.6
sparsity_num_format=bfp
benchmark=cifar10
rearrange=False

N="[-2,-2]"
M="[-2,-2]"

unconstrained=True
bit_range="[2,5]"

   if [ $sparsity_num_format == "fp32" ]
   then
      filename=$sparsity_num_format/fp32\_$N:$M
   else
      filename=$sparsity_num_format\_block\_size\_$blocksize/hbfp\_$bit_range/$benchmark\_bfp$mantbits\_sparse\_$blocksize
      mkdir ./sparse_results/$benchmark/quant_scheme1/$sparsity_num_format\_block\_size\_$blocksize/
      mkdir ./sparse_results/$benchmark/quant_scheme1/$sparsity_num_format\_block\_size\_$blocksize/hbfp\_$bit_range/
   fi

   if [ $compute_node == "runai" ]
   then
      rm /usr/local/lib/python3.8/dist-packages/transformers/bfp/bfp_config.yaml
      rm /usr/local/lib/python3.8/dist-packages/transformers/bfp/bfp_ops.py
      cp ../../../src/transformers/bfp/bfp_ops.py /usr/local/lib/python3.8/dist-packages/transformers/bfp/
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
   num_format: 'bfp'
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
   cd examples/pytorch/image-classification/
   python3 run_image_classification.py  \
      --dataset_name $benchmark  \
      --output_dir ./sparse_results/$benchmark/quant_scheme1/$filename  \
      --overwrite_output_dir \
      --remove_unused_columns False  \
      --do_train  \
      --do_eval  \
      --learning_rate 5e-5  \
      --num_train_epochs 3  \
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
      --optim BFPAdam | tee ./sparse_results/$benchmark/quant_scheme1/$filename.txt