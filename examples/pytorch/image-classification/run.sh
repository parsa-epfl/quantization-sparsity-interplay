for datatype in fp32 bfp
do
   if [ $datatype$ == "bfp" ]
   then
      for mantbit in 3 5 7
      do
         rm ../../../src/transformers/bfp/bfp_config.yaml
         echo -e "hbfp: \n\t num_format: 'bfp' \n\t rounding_mode: 'stoc' \n\t epsilon: 0.00000001 \n\t mant_bits: $mantbit \n\t weight_mant_bits: 15 \n\t bfp_tile_size: 8 \n\t bfp_block_size: 64 \n\t in_sparsity: False \n\t w_sparsity: True \n\t grad_sparsity: False \n\t device: 'cuda'" >> ../../../src/transformers/bfp/bfp_config.yaml
         python3 run_image_classification.py  \
            --dataset_name cifar100  \
            --output_dir ./cifar100_hbfp$manbit\_sparse,64/  \
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
            --optim sgd | tee cifar100_hbfp$manbit\_sparse,64.txt

         rm ../../../src/transformers/bfp/bfp_config.yaml
         echo -e "hbfp: \n\t num_format: 'bfp' \n\t rounding_mode: 'stoc' \n\t epsilon: 0.00000001 \n\t mant_bits: $mantbit \n\t weight_mant_bits: 15 \n\t bfp_tile_size: 8 \n\t bfp_block_size: 64 \n\t in_sparsity: False \n\t w_sparsity: False \n\t grad_sparsity: False \n\t device: 'cuda'" >> ../../../src/transformers/bfp/bfp_config.yaml
         python3 run_image_classification.py  \
            --dataset_name cifar100  \
            --output_dir ./cifar100_hbfp$manbit,64/  \
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
            --optim sgd | tee cifar100_hbfp$manbit,64.txt
      done
   else
      rm ../../../src/transformers/bfp/bfp_config.yaml
      echo -e "hbfp: \n\t num_format: 'fp32' \n\t rounding_mode: 'stoc' \n\t epsilon: 0.00000001 \n\t mant_bits: 7 \n\t weight_mant_bits: 15 \n\t bfp_tile_size: 8 \n\t bfp_block_size: 64 \n\t in_sparsity: False \n\t w_sparsity: True \n\t grad_sparsity: False \n\t device: 'cuda'" >> ../../../src/transformers/bfp/bfp_config.yaml
      python3 run_image_classification.py  \
            --dataset_name cifar100  \
            --output_dir ./cifar100_fp32_sparse/  \
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
            --optim sgd | tee cifar100_fp32_sparse.txt

      rm ../../../src/transformers/bfp/bfp_config.yaml
      echo -e "hbfp: \n\t num_format: 'fp32' \n\t rounding_mode: 'stoc' \n\t epsilon: 0.00000001 \n\t mant_bits: 7 \n\t weight_mant_bits: 15 \n\t bfp_tile_size: 8 \n\t bfp_block_size: 64 \n\t in_sparsity: False \n\t w_sparsity: False \n\t grad_sparsity: False \n\t device: 'cuda'" >> ../../../src/transformers/bfp/bfp_config.yaml
      python3 run_image_classification.py  \
            --dataset_name cifar100  \
            --output_dir ./cifar100_fp32/  \
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
            --optim sgd | tee cifar100_fp32.txt
   fi
done