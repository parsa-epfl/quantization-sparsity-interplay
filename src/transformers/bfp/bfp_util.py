import yaml
import os

def get_bfp_args():
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    with open(os.path.join(__location__, 'bfp_config.yaml')) as file:
        try:
            hbfpConfig = yaml.safe_load(file)
            print(hbfpConfig['hbfp'])   
            return hbfpConfig['hbfp']
        except yaml.YAMLError as exc:
            print(exc)

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
    mx_args["block_size"] = bfp_args["bfp_block_size"]
    mx_args["bfloat"] = bfp_args["bfloat"]
    mx_args["scale_bits"] = bfp_args["scale_bits"] 
    return mx_args

if __name__ == '__main__':
    get_bfp_args()
