blocksize=2
mantbits=7
sparsity_frac=0.65
sparsity_num_format=fp32
benchmark=cifar10
rearrange=False
# for sparsity_frac in 0.7 0.8 0.4
# do
# for blocksize in 2 4 8 16 32
# do
# for mantbits in 6 4
# do
   if [ $sparsity_num_format == "fp32" ]
   then
      filename=$sparsity_num_format/$sparsity_frac\_percent
   else
      filename=$sparsity_num_format\_block\_size\_$blocksize/$sparsity_frac\_percent/$benchmark\_bfp$mantbits\_sparse\_$blocksize
      mkdir ./sparse_results/$benchmark/sparsity_scheme1/$sparsity_num_format\_block\_size\_$blocksize/
      mkdir ./sparse_results/$benchmark/sparsity_scheme1/$sparsity_num_format\_block\_size\_$blocksize/$sparsity_frac\_percent/
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
   w_sparsity: True 
   grad_sparsity: False
   rearrange: $rearrange
   sparsity_frac: $sparsity_frac 
   device: 'cuda'" >> ../../../src/transformers/bfp/bfp_config.yaml
   cd ../../../
   pip install -e .
   cd examples/pytorch/image-classification/
   python3 run_image_classification.py  \
      --dataset_name $benchmark  \
      --output_dir ./sparse_results/$benchmark/sparsity_scheme1/$filename  \
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
      --optim BFPAdam | tee ./sparse_results/$benchmark/sparsity_scheme1/$filename.txt
# done
# done
# done