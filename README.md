# Effective Interplay between Sparsity and Quantization: From Theory to Practice

This repository is the official implementation of the code used for all analysis and experiments in the paper: Effective Interplay between Sparsity and Quantization: From Theory to Practice.

The paper mathematically investigates the relationship between quantization and sparsity techniques, and how their errors combine when both techniques are used together. The theoretical analysis is validated by experimental results on a wide range of models.

## About Our Work
Various forms of quantization and sparsity techniques have emerged as promising approaches to compress models, especially in the modern era of LLMs. This paper focuses on the combined application of both of these techniques, and is part of the broader research efforts to make the memory footprint of LLMs smaller, and make them more accessible. Our mathematical analysis and extensive empirical study with large language models (OPT, LLaMA) and vision transformers (ViT) demonstrate that quantization and sparsity are not orthogonal and their combined use can adversely affect model accuracy. Our findings provide valuable insights for optimizing the compression of large models while preserving accuracy.


To setup the environment, please run:
```console
pip install -r requirements_pip.txt
```

Scripts to run LLaMA, OPT and ViT experiments are provided.

Access scripts for LLaMA and OPT in the following directory:
```console
cd ./examples/pytorch/language-modeling/nips_configs/
```

Access scripts for ViT in the following directory:
```console
cd ./examples/pytorch/image-classification/
```

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
