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

This project focuses on implementing Swin on an image classification task and shows that with modifications, supervised training of the Swin transformer model on small scale datasets like `CIFAR-10` can lead to very high accuracy with low computational constraints. The Window Attention block can be seen here:


```python
class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2 Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn_out = attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_out
```

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

Model summary:
```
==================================================================================================
Layer (type:depth-idx)                             Kernel Shape     Output Shape     Param #
==================================================================================================
SwinTransformer                                    --               [1, 10]          --
├─PatchEmbed: 1-1                                  --               [1, 256, 96]     --
│    └─Conv2d: 2-1                                 [2, 2]           [1, 96, 16, 16]  1,248
│    └─LayerNorm: 2-2                              --               [1, 256, 96]     192
├─Dropout: 1-2                                     --               [1, 256, 96]     --
├─ModuleList: 1-3                                  --               --               --
│    └─BasicLayer: 2-3                             --               [1, 64, 192]     --
│    │    └─ModuleList: 3-1                        --               --               149,862
│    │    └─PatchMerging: 3-2                      --               [1, 64, 192]     74,496
│    └─BasicLayer: 2-4                             --               [1, 16, 384]     --
│    │    └─ModuleList: 3-3                        --               --               1,783,908
│    │    └─PatchMerging: 3-4                      --               [1, 16, 384]     296,448
│    └─BasicLayer: 2-5                             --               [1, 16, 384]     --
│    │    └─ModuleList: 3-5                        --               --               4,737,840
├─LayerNorm: 1-4                                   --               [1, 16, 384]     768
├─AdaptiveAvgPool1d: 1-5                           --               [1, 384, 1]      --
├─Linear: 1-6                                      --               [1, 10]          3,850
==================================================================================================
Total params: 7,048,612
Trainable params: 7,048,612
Non-trainable params: 0
Total mult-adds (M): 11.15
==================================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 11.50
Params size (MB): 28.18
Estimated Total Size (MB): 39.69
==================================================================================================
```

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
