# Int4 and sparsity experiments

This branch contains the extension of our Quantization-and-Sparsity Emulator focusing on various integer quantization and structured sparsity combinations. This mode is currently supported for LLMs from OPT family.

In order to access the official version of Emulator used for all analysis and experiments in the paper: Effective Interplay between Sparsity and Quantization: From Theory to Practice, switch to the `main` branch.

To setup the environment, please run:
```console
pip install -r requirements_pip.txt
```

To run the experiments, visit the following directory:
```console
cd ./int4-sparse-experiments
```

There you'll find:
- example scripts for sparse-and-quantized fine-tuning and inference 
- example script to evaluate layerwise sensitivity
- instructions how to customize configurations for supported models
- instructions how to enable sparsity and quantization for any LLM


## Citation
If you find the analysis and experimental results useful for your own research, please cite our paper:
```angular2html
@article{quant-sparse-interplay:2024,
    title        = {{Effective Interplay between Sparsity and Quantization:
From Theory to Practice}},
    author       = {Harma, Simla Burcu and Chakraborty, Ayan and Kostenok, Elizaveta and Mishin, Danila and Ha, Dongho and Falsafi, Babak and Jaggi, Martin and Liu, Ming and Oh, Yunho and Subramanian, Suvinay and Yazdanbakhsh, Amir},
    year         = 2024,
    journal      = {arXiv preprint}
}
```
