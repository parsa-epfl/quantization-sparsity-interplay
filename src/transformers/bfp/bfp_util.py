import yaml
import os

def unpack_bfp_args(kwargs={}):
    bfp_args = {}
    bfp_argn = [('sparsity_num_format', 'fp'),
                ('a_num_format', 'fp'),
                ('a_mant_bits', 32),
                ('w_num_format', 'fp'),
                ('w_mant_bits', 32),
                ('rounding_mode', 'stoc'),
                ('epsilon', 1e-8),
                ('block_size', 0),
                ('in_sparsity', False),
                ('w_sparsity', False),
                ('grad_sparsity', False),
                ('N', 0), 
                ('M', 0),
                ('first', 's'),
                ('sparsity_mode', 'unstructured'),
                ('sparsity_frac', 0.0),
                ('sparsify', False),
                # ('mx_w_elem_format', ''),
                # ('mx_a_elem_format', ''),
                # ('bfloat', 16),
                # ('scale_bits', 8),
                ('device', 'cpu')]

    for arg, default in bfp_argn:
        if arg in kwargs:
            bfp_args[arg] = kwargs[arg]
            del kwargs[arg]
        else:
            bfp_args[arg] = default
    return bfp_args

def get_bfp_args(filename='bfp_config.yaml'):
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    with open(os.path.join(__location__, filename)) as file:
        try:
            hbfpConfig = yaml.safe_load(file)
            print(hbfpConfig['hbfp'])
            return hbfpConfig['hbfp']
        except yaml.YAMLError as exc:
            print(exc)

def get_bfp_args_per_layer(layer_type, layer_idx):
    full_config = get_bfp_args(filename='bfp_layer_config.yaml')
    layer_quant_args = full_config[layer_type]

    if layer_idx in layer_quant_args['layer_ids']:
        # set sparse-and-quantized configuration for target layers
        layer_config = unpack_bfp_args(layer_quant_args)
    else:
        # set fp32 dense configuration for remaining layers
        layer_config = unpack_bfp_args()

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
