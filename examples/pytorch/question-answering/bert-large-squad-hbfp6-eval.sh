compute_node=$1
blocksize=64
mantbits=5
sparsity_frac=0.6
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
      mkdir /home/parsa_liza/experiments/bert_large_squad_hbfp6_sparse_eval/$benchmark/quant_scheme2/$sparsity_num_format\_block\_size\_$blocksize/
      mkdir /home/parsa_liza/experiments/bert_large_squad_hbfp6_sparse_eval/$benchmark/quant_scheme2/$sparsity_num_format\_block\_size\_$blocksize/hbfp\_$bit_range/
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
   device: 'cuda:0'" >> /usr/local/lib/python3.8/dist-packages/transformers/bfp/bfp_config.yaml
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
   device: 'cuda:0'" >> ../../../src/transformers/bfp/bfp_config.yaml
      cd ../../../
      pip install -e .
   fi
cd examples/pytorch/question-answering
CUDA_VISIBLE_DEVICES=1 python run_qa.py \
    --model_name_or_path /home/parsa_liza/experiments/bert_large_squad_hbfp6_sparse/quant_scheme2/checkpoint-15500 \
    --dataset_name squad \
    --do_eval \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir /home/parsa_liza/experiments/bert_large_squad_hbfp6_sparse_eval/$benchmark/quant_scheme2/$filename  \
