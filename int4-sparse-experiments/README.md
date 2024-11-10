# Examples

Current implementation supports **INT-N** weight quantization, **FP16/8** activation quantization and structured **N:M** sparsity applied to **Feed-Forward** layers (`FF1`, `FF2` for OPT model family). It allows assigning different sparse configurations and bitwidths to layers at various locations in the network.

We provide the following example configurations in the folder `scripts`:
1. `opt-6.7b-finetuning.sh` set-ups sparse-and-quantized fine-tuning for OPT-6.7b
- **Quantization**: 
    - INT4 weights and FP16 activations for the specified layers
    - FP32 weights and activations for the remaining layers 
- **Sparsity**: 
    - FF1 2:4 and FF2 1:4 sparsity for the specified layers
    - remaining layers are dense
2. `opt-6.7b-inference.sh` applies sparsity and quantization to OPT-6.7b in zero-shot manner without fine-tuning
- **Quantization**: 
    - INT4 weights and FP16 activations for all layers
- **Sparsity**: 
    - FF1 2:4 and FF2 1:4 sparsity for the robust layers (`configuration_1`)
    - sensitive layers are dense (`configuration_2`) 
3. `opt-6.7b-layerwise-sensitivity.sh` iterates through attention modules of OPT-6.7b and evaluates the impact of sparsity and quantization applied to a single layer in zero-shot manner
- **Quantization**: 
  - INT4 weights and FP16 activations for the single specified layer
  - FP32 weights and activations for the remaining layers 
- **Sparsity**: 
    - FF1 2:4 and FF2 1:4 sparsity for the single specified layer
    - remaining layers are dense
4. `opt-125m-inference.sh` applies sparsity and quantization to OPT-125m in zero-shot manner (without fine-tuning)
- **Quantization**: 
    - INT4 weights and FP16 activations for the specified layers
    - FP32 weights and activations for the remaining layers 
- **Sparsity**: 
    - FF1 2:4 and FF2 2:4 sparsity for the specified layers
    - remaining layers are dense


**Evaluation**: Perplexity on the WikiText dataset is calculated and saved to the file `OUTPUT_DIR\results.txt` in the experiments folder. Per-layer sparse-and-quantized configurations are dumped to the same file.

# Customizing provided configurations for OPT models

You can specify the compression configuration for the LAYER_TYPE (e.g. `fc1`, `fc2` for OPT models) at particular indices using the following template:

```console
LAYER_TYPE:                               
  - layer_ids: [LAYER_1, LAYER_2, ...]    # specify all layer indices to apply the configuration below
    sparsity_num_format: bfp              # do not modify this line
    a_num_format: fp                      # numerical format of activations
    a_mant_bits: 16                       # activation bitwidth
    w_num_format: int                     # numerical format of weights
    w_mant_bits: 4                        # weight bitwidth
    block_size: 32                        # block size for integer quantization
    w_sparsity: True                      # whether to sparsify weights 
    N: 2                                  # hyperparameter of structured sparsity 
    M: 4                                  # hyperparameter of structured sparsity
    first: s                              # do not modify this line 
    sparsity_frac: 0.5                    # sparsity fraction for unstructured sparsity
    sparsity_mode: structured             # sparsity mode
    device: cuda
    sparsify: True                        # whether to sparsify layer, should match 'w_sparsity' parameter
```
If sparse-and-quantized configuration for a layer is not specified, it remains dense and in full precision.

Note: `LAYER_TYPE` in the configuration should match the `layer_type` argument of `bfp_util.get_bfp_args_per_layer` function called in the model source file (e.g. `transformers_hbfp_sparsity/src/transformers/models/opt/modeling_opt.py`, lines 307-308)

# Enabling sparsity and quantization for other model families, e.g. LLaMA, Mistral
Current implementation supports sparse-and-quantized fine-tuning of OPT models. In order to enable compression for any layer of any LLM, please replace the corresponding `nn.Linear` layers with the `BFPLinear` module in the model source file.

For example, compressing `q_proj` sublayer in the attention layers of the LLaMA model requires the following modifications in the `transformers_hbfp_sparsity/src/transformers/models/llama/modeling_llama.py`:
1. Import the BFP functions and modules:
```console
from ...bfp import bfp_util
from ...bfp.bfp_ops import BFPLinear
```
2. In the `LlamaAttention` module, extract the sparse-and-quantized configuration for the layer and replace the corresponding layer:
```console
self.bfp_q_proj_args = bfp_util.get_bfp_args_per_layer(layer_type="q_proj", layer_idx=self.layer_idx)
self.q_proj = BFPLinear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias, **bfp_q_proj_args)
```
You may need to add the layer index as the attribute of the module. Please refer to the example in `transformers_hbfp_sparsity/src/transformers/models/opt/modeling_opt.py`
