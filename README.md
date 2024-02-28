# Swin Transformer on CIFAR-10

<hr>

## Contents

1. [Highlights](#Highlights)
2. [Requirements](#Requirements)
3. [Usage](#Usage)
4. [Results](#Results)


<hr>

## Highlights
This project is a implementation from scratch of a slightly modified version of the Swin transformer introduced in the paper [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030). We implement this model on the small scale benchmark dataset `CIFAR-10`. 

**Swin Transformer** (the name `Swin` stands for **S**hifted **win**dow) is initially described in [arxiv](https://arxiv.org/abs/2103.14030), which capably serves as a general-purpose backbone for computer vision. It is basically a hierarchical Transformer whose representation is computed with shifted windows. The shifted windowing scheme brings greater efficiency by limiting self-attention
computation to non-overlapping local windows while also allowing for cross-window connection.

The proposed Swin Transformer builds hierarchical feature maps by merging image patches (shown in gray) in deeper layers and has linear computation complexity to input image size due to computation of self-attention only within each local window (shown in red). It can thus serve as a general-purpose backbone for both image classification and dense recognition tasks. In contrast, previous vision Transformers produce feature maps of a single low resolution and have quadratic computation complexity to input image size due to computation of self-attention globally.

<img src="./images/swin1.png" width="550"></img>

Swin departs from the traditional method of computing self-attention and implements a shifted window approach for computing self-attention. In layer $l$ (left), a regular window partitioning scheme is adopted, and self-attention is computed within each window. In the next layer $l+1$ (right), the window partitioning is shifted, resulting in new windows. The self-attention computation in the new windows crosses the boundaries of the previous windows in layer $l$, providing connections among them. 

<img src="./images/swin2.png" width="550"></img>

The overall architecture of the Swin model can be seen below:

<img src="./images/swin3.png" width="850"></img>

This project focuses on implementing Swin on an image classification task and shows that with modifications, supervised training of the Swin transformer model on small scale datasets like `CIFAR-10` can lead to very high accuracy with low computational constraints.

<hr>

## Requirements
```shell
pip install -r requirements.txt
```

<hr>

## Usage
To replicate the reported results, clone this repo
```shell
cd your_directory git clone git@github.com:jordandeklerk/SwinViT-pytorch.git
```
and run the main training script
```shell
python train.py 
```
Make sure to adjust the checkpoint directory in `train.py` to store checkpoint files.

<hr>

## Results
We test our approach on the `CIFAR-10` dataset with the intention to extend our model to 4 other small low resolution datasets: `Tiny-Imagenet`, `CIFAR100`, `CINIC10` and `SVHN`. All training took place on a single A100 GPU.
  * CIFAR10
    * ```swin_cifar10_patch2_input32``` - 91.10 @ 32

Flop analysis:
```
total flops: 242759424
total activations: 1296394
number of parameter: 7048612
| module       | #parameters or shape   | #flops   |
|:-------------|:-----------------------|:---------|
| model        | 7.049M                 | 0.243G   |
|  patch_embed |  1.44K                 |  0.418M  |
|  layers      |  7.043M                |  0.242G  |
|  norm        |  0.768K                |  30.72K  |
|  head        |  3.85K                 |  3.84K   |
```

<hr>

## Citation
```bibtex
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```
