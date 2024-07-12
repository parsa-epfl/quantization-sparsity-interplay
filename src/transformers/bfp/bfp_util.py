import yaml
import os

def get_bfp_args(filename='bfp_config.yaml'):
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    with open(os.path.join(__location__, filename)) as file:
        try:
            hbfpConfig = yaml.safe_load(file)
            print(hbfpConfig['hbfp'])
            return hbfpConfig['hbfp']
        except yaml.YAMLError as exc:
            print(exc)

def get_bfp_args_per_layer(layer_type="attn", layer_idx=self.layer_idx):
    full_config = get_bfp_args(filename='bfp_layer_config.yaml')
    layer_config = full_config[layer_type]
    return layer_config


def extract_sparsity_args(bfp_args):
    sparsity_args = {}
    if bfp_args["w_sparsity"]:
        sparsity_args["sparsity"] = True
    else:
        sparsity_args["sparsity"] = False
    for arg_name in ["device", "sparsity_mode", "sparsity_frac", "N", "M"]:
        sparsity_args[arg_name] = bfp_args[arg_name]
    return sparsity_args

def extract_mx_args(bfp_args):
    mx_args = {}
    mx_args["w_elem_format"] = bfp_args["mx_w_elem_format"]
    mx_args["a_elem_format"] = bfp_args["mx_a_elem_format"]
    mx_args["block_size"] = bfp_args["block_size"]
    mx_args["bfloat"] = bfp_args["bfloat"]
    mx_args["scale_bits"] = bfp_args["scale_bits"] 
    return mx_args

if __name__ == '__main__':
    get_bfp_args()
