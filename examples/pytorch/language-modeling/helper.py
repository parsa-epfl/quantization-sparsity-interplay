import numpy as np
import torch
# ppl = np.load("/parsadata1/lisa/experiments/magn_based/125m/ppl.npy")
# print(ppl)
new_dict = {}
model_st_dict = torch.load("/parsadata1/lisa/experiments/magn_based/6.7b/fp_2:4/full_model.pth")
for key in model_st_dict:
    # print(key, model_st_dict[key].shape)
    if key != "lm_head.weight":
        new_dict[key] = model_st_dict[key]
torch.save(new_dict, "/parsadata1/lisa/experiments/magn_based/6.7b/fp_2:4/full_model_no_lm_head.pth")
# print(model_st_dict["lm_head.weight"].shape)
# print(list(model_st_dict.keys()))