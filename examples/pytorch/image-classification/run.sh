for datatype in fp32 bfp
do
   if [ $datatype == "bfp" ]
   then
      for mantbit in 3 5 7
      do
         rm ../../../src/transformers/bfp/bfp_config.yaml
         echo -e "hbfp:
   num_format: 'bfp' 
   rounding_mode: 'stoc' 
   epsilon: 0.00000001 
   mant_bits: $mantbit 
   weight_mant_bits: 15 
   bfp_tile_size: 8 
   bfp_block_size: 64 
   in_sparsity: False
   w_sparsity: True 
   grad_sparsity: False 
   device: 'cuda'" >> ../../../src/transformers/bfp/bfp_config.yaml
         python3 run_image_classification.py  \
            --dataset_name cifar10  \
            --output_dir ./sparse_results/cifar10_hbfp$mantbit\_sparse,64/  \
            --overwrite_output_dir \
            --remove_unused_columns False  \
            --do_train  \
            --do_eval  \
            --learning_rate 5e-5  \
            --num_train_epochs 10  \
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
            --optim sgd | tee ./sparse_results/cifar10_hbfp$mantbit\_sparse,64.txt

         rm ../../../src/transformers/bfp/bfp_config.yaml
         echo -e "hbfp:
   num_format: 'bfp' 
   rounding_mode: 'stoc' 
   epsilon: 0.00000001 
   mant_bits: $mantbit 
   weight_mant_bits: 15 
   bfp_tile_size: 8 
   bfp_block_size: 64 
   in_sparsity: False 
   w_sparsity: False 
   grad_sparsity: False 
   device: 'cuda'" >> ../../../src/transformers/bfp/bfp_config.yaml
         python3 run_image_classification.py  \
            --dataset_name cifar10  \
            --output_dir ./sparse_results/cifar10_hbfp$mantbit,64/  \
            --overwrite_output_dir \
            --remove_unused_columns False  \
            --do_train  \
            --do_eval  \
            --learning_rate 5e-5  \
            --num_train_epochs 10  \
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
            --optim sgd | tee ./sparse_results/cifar10_hbfp$mantbit,64.txt
      done
   else
      rm ../../../src/transformers/bfp/bfp_config.yaml
      echo -e "hbfp: 
   num_format: 'fp32' 
   rounding_mode: 'stoc' 
   epsilon: 0.00000001 
   mant_bits: 7 
   weight_mant_bits: 15 
   bfp_tile_size: 8 
   bfp_block_size: 64 
   in_sparsity: False 
   w_sparsity: True 
   grad_sparsity: False 
   device: 'cuda'" >> ../../../src/transformers/bfp/bfp_config.yaml
      python3 run_image_classification.py  \
            --dataset_name cifar10  \
            --output_dir ./sparse_results/cifar10_fp32_sparse/  \
            --overwrite_output_dir \
            --remove_unused_columns False  \
            --do_train  \
            --do_eval  \
            --learning_rate 5e-5  \
            --num_train_epochs 10  \
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
            --optim sgd | tee ./sparse_results/cifar10_fp32_sparse.txt

      rm ../../../src/transformers/bfp/bfp_config.yaml
      echo -e "hbfp: 
   num_format: 'fp32'
   rounding_mode: 'stoc' 
   epsilon: 0.00000001 
   mant_bits: 7 
   weight_mant_bits: 15 
   bfp_tile_size: 8 
   bfp_block_size: 64 
   in_sparsity: False
   w_sparsity: False 
   grad_sparsity: False 
   device: 'cuda'" >> ../../../src/transformers/bfp/bfp_config.yaml
      python3 run_image_classification.py  \
            --dataset_name cifar10  \
            --output_dir ./sparse_results/cifar10_fp32/  \
            --overwrite_output_dir \
            --remove_unused_columns False  \
            --do_train  \
            --do_eval  \
            --learning_rate 5e-5  \
            --num_train_epochs 10  \
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
            --optim sgd | tee ./sparse_results/cifar10_fp32.txt
   fi
done