model='facebook/opt-6.7b'
filename=opt-6.7\_sensitivity-exp\_chkpt


rm ../../../../src/transformers/bfp/bfp_layer_config.yaml
echo -e "hbfp:
    fc1:
        conf_1:
            layer_ids: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 17, 22, 23, 24, 26]
            sparsity_num_format: bfp
            a_num_format: fp
            a_mant_bits: 16
            w_num_format: int
            w_mant_bits: 4
            block_size: 32 
            w_sparsity: True
            N: 2
            M: 4
            first: s
            sparsity_mode: structured
            device: cuda
            sparsify: True
            sparsity_frac: 0.5
        conf_2:
            layer_ids: [1, 2, 13, 14, 16, 18, 19, 20, 21, 25, 27, 28, 29, 30]
            sparsity_num_format: bfp
            a_num_format: fp
            a_mant_bits: 16
            w_num_format: int
            w_mant_bits: 4
            block_size: 32 
            w_sparsity: False
            N: 2
            M: 4
            first: s
            sparsity_mode: structured
            device: cuda
            sparsify: False
            sparsity_frac: 0.5
    fc2:
        conf_1:
            layer_ids: [1, 2, 13, 14, 16, 18, 19, 20, 21, 25, 27, 28, 29, 30]
            sparsity_num_format: bfp
            a_num_format: fp
            a_mant_bits: 16
            w_num_format: int
            w_mant_bits: 4
            block_size: 32 
            w_sparsity: False
            N: 1
            M: 4
            first: s
            sparsity_mode: structured
            device: cuda
            sparsify: False
            sparsity_frac: 0.5
        conf_2:   
            layer_ids: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 17, 22, 23, 24, 26]
            sparsity_num_format: bfp
            a_num_format: fp
            a_mant_bits: 16
            w_num_format: int
            w_mant_bits: 4
            block_size: 32 
            w_sparsity: True
            N: 1
            M: 4
            first: s
            sparsity_mode: structured
            device: cuda
            sparsify: True
            sparsity_frac: 0.5" >> ../../../../src/transformers/bfp/bfp_layer_config.yaml

cd ../../../../
pip install -e .

cd examples/pytorch/language-modeling
python3 run_opt.py \
    --model_name_or_path $model \
    --tokenizer_name $model \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --do_eval \
    --remove_unused_columns False \
    --output_dir $filename \
    --overwrite_output_dir \
    --learning_rate 5e-04 \
    --adam_beta1 0.9  \
    --adam_beta2 0.999  \
    --adam_epsilon 1e-08  \
    --lr_scheduler_type linear \
    --max_steps 1000 \
