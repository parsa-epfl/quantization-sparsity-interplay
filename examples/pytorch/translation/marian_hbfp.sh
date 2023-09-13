if [ $# -lt 1 ]; then
  echo 1>&2 "$0: not enough arguments"
  exit 2
fi
compute_node=$1
blocksize=64
mantbits=7
sparsity_frac=0.6
sparsity_num_format=bfp
benchmark=ted_iwlst2013
rearrange=False

N="[-2,-2]"
M="[-2,-2]"

unconstrained=True
bit_range="[2,3]"

   if [ $sparsity_num_format == "fp32" ]
   then
      filename=$sparsity_num_format/fp32\_$N:$M
   else
      filename=$sparsity_num_format\_block\_size\_$blocksize/hbfp\_$bit_range/$benchmark\_bfp$mantbits\_sparse\_$blocksize
      mkdir /home/parsa_liza/experiments/marian_12.09/$benchmark/quant_scheme2/$sparsity_num_format\_block\_size\_$blocksize/
      mkdir /home/parsa_liza/experiments/marian_12.09/$benchmark/quant_scheme2/$sparsity_num_format\_block\_size\_$blocksize/hbfp\_$bit_range/
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
   w_sparsity: True 
   grad_sparsity: False
   rearrange: $rearrange
   sparsity_frac: $sparsity_frac
   N: $N
   M: $M 
   unconstrained: $unconstrained
   bit_range: $bit_range
   device: 'cuda:1'" >> /usr/local/lib/python3.8/dist-packages/transformers/bfp/bfp_config.yaml
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
   w_sparsity: True
   grad_sparsity: False
   rearrange: $rearrange
   sparsity_frac: $sparsity_frac
   N: $N
   M: $M
   unconstrained: $unconstrained
   bit_range: $bit_range
   device: 'cuda:1'" >> ../../../src/transformers/bfp/bfp_config.yaml
      cd ../../../
      pip install -e .
   fi
   cd examples/pytorch/translation/
   python3 run_translation_marian.py  \
      --model_name_or_path Helsinki-NLP/opus-mt-de-en \
      --source_lang de \
      --target_lang en \
      --dataset_name $benchmark  \
      --dataset_config_name de-en \
      --output_dir /home/parsa_liza/experiments/marian_12.09/$benchmark/quant_scheme2/$filename  \
      --overwrite_output_dir \
      --remove_unused_columns False  \
      --do_train  \
      --do_eval  \
      --learning_rate 5e-5  \
      --num_train_epochs 1 \
      --per_device_train_batch_size 4  \
      --per_device_eval_batch_size 4  \
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
      --optim BFPAdam | tee /home/parsa_liza/experiments/marian_12.09/$benchmark/quant_scheme2/$filename.txt