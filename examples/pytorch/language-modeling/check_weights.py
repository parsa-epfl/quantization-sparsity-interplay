import torch

state_dict = torch.load("/home/parsa_liza/experiments/bert_better_init_10ep/quant_scheme2/bfp_block_size_64/hbfp_[]/_bfp7_sparse_64/pytorch_model.bin")
state_dict_fp = torch.load("/home/parsa_liza/experiments/bert_fp_dense/quant_scheme2/fp32/fp32_[2]:[4]/pytorch_model.bin")
for i in range(11):
    param_name = "bert.encoder.layer." + str(i) + ".attention.self.key.weight"
    print(state_dict[param_name][0:8, 0])
    print(state_dict_fp[param_name][0:8, 0])
