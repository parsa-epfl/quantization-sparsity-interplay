model='facebook/opt-125m'
output_dir=opt-125m\_chkpt


rm ../../src/transformers/bfp/bfp_layer_config.yaml
echo -e "hbfp:
    fc1:
        - layer_ids: [6, 7, 8, 9, 10, 11]
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
    fc2:   
        - layer_ids: [6, 7, 8, 9, 10, 11]
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
          sparsity_frac: 0.5" >> ../../src/transformers/bfp/bfp_layer_config.yaml

cd ../../
pip install -e .

cd int4-sparse-experiments
python3 run_opt.py \
    --model_name_or_path $model \
    --tokenizer_name $model \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --seed 14 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_eval \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --learning_rate 1e-04 \
    --adam_beta1 0.9  \
    --adam_beta2 0.999  \
    --adam_epsilon 1e-08  \
    --lr_scheduler_type linear \
    --num_train_epochs 3 \
