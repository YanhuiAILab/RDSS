# Enhancing Semi-Supervised Learning via Representative and Diverse Sample Selection (NeurIPS, 2024)

by Qian Shao<sup>1</sup>, Jiangrui Kang<sup>2</sup>, Qiyuan Chen<sup>1</sup>, Zepeng Li<sup>1</sup>, Hongxia Xu<sup>1</sup>, Yiwen Cao<sup>2</sup>, Jiajuan Liang<sup>2</sup>, Jian Wu<sup>1</sup>

<sup>1</sup>Zhejiang University, <sup>2</sup>BNU-HKBU United International College

This is the official implementation of [Enhancing Semi-Supervised Learning via Representative and Diverse Sample Selection (RDSS)](https://arxiv.org/abs/2409.11653) in PyTorch.

## Prerequisites

To install the required packages, you can create a conda environment:

```sh
conda create --name RDSS python=3.8
```

then use pip to install required packages:

```sh
pip install -r requirements.txt
```

## Download

First, download feature-extracting model [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)

Next, prepare the datasets and store them in the `data` folder.

## Start Sampling

From now on, you can extract features and select samples by typing

```sh
python get_cifar_feature.py
```

```sh
python RDSS.py
```

## Citation

If you find our paper and repo useful, please cite our paper.

```
@misc{shao2024enhancingsemisupervisedlearningrepresentative,
      title={Enhancing Semi-Supervised Learning via Representative and Diverse Sample Selection}, 
      author={Qian Shao and Jiangrui Kang and Qiyuan Chen and Zepeng Li and Hongxia Xu and Yiwen Cao and Jiajuan Liang and Jian Wu},
      year={2024},
      eprint={2409.11653},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.11653}, 
}
```
