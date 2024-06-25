# Swin Transformer on CIFAR-10

## Highlights

<img src="./images/swin3.png"></img>

This project is an implementation of a slightly modified version of the Swin transformer introduced in the paper [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030). We implement this model on the small scale benchmark dataset `CIFAR-10`. 

**Swin Transformer** (the name `Swin` stands for **S**hifted **win**dow) is initially described in [arxiv](https://arxiv.org/abs/2103.14030), which capably serves as a general-purpose backbone for computer vision. It is basically a hierarchical Transformer whose representation is computed with shifted windows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection.

This project focuses on implementing Swin on an image classification task and shows that with modifications, supervised training of the Swin transformer model on small scale datasets like `CIFAR-10` can lead to very high accuracy with low computational constraints.

## Project Structure

```
├── main.py
├── model
│   └── swin_vit.py
├── requirements.txt
└── utils
    ├── autoaug.py
    ├── cutmix.py
    ├── dataloader.py
    ├── loss.py
    ├── optimizer.py
    ├── parser.py
    ├── random_erasing.py
    ├── sampler.py
    ├── scheduler.py
    ├── train_functions.py
    ├── transforms.py
    └── utils.py
```

## Usage

### Install Dependencies

Create a virtual environment and clone this repository:

```bash
# Clone the repository
git clone https://github.com/jordandeklerk/SwinViT.git
cd SwinViT

# Create a virtual environment
python3 -m venv myenv

# Activate the virtual environment
source myenv/bin/activate

# Install the required Python packages
pip install -r requirements.txt
```

<hr>

## Usage
To replicate the reported results, run `main.py` with the following hyperparameters:

```bash
python main.py  --patch_size 2 \
                --weight_decay 0.1 \
                --batch_size 128 \
                --epochs 200 \
                --lr 0.001 \
                --warmup_epochs 10 \
                --min_lr 1e-6 \
                --clip_grad 3.0 
```

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

## Citation
```bibtex
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```
