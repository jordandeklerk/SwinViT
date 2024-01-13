import json, os, math
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F
from typing import Tuple
import torchvision
from torchvision import transforms

from functools import partial
from typing import Optional, Callable
from contextlib import nullcontext, contextmanager

from torch.nn import Module

from accelerate import Accelerator
from accelerate.tracking import WandBTracker


def save_experiment(experiment_name, config, model, train_losses, test_losses, accuracies, base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    
    # Save the config
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)
    
    # Save the metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'w') as f:
        data = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'accuracies': accuracies,
        }
        json.dump(data, f, sort_keys=True, indent=4)
    
    # Save the model
    save_checkpoint(experiment_name, model, "final", base_dir=base_dir)


def save_checkpoint(experiment_name, model, epoch, base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    cpfile = os.path.join(outdir, f'model_{epoch}.pt')
    torch.save(model.state_dict(), cpfile)


def load_experiment(experiment_name, checkpoint_name="model_final.pt", base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    # Load the config
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'r') as f:
        config = json.load(f)
    # Load the metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'r') as f:
        data = json.load(f)
    train_losses = data['train_losses']
    test_losses = data['test_losses']
    accuracies = data['accuracies']
    # Load the model
    model = model(config)
    cpfile = os.path.join(outdir, checkpoint_name)
    model.load_state_dict(torch.load(cpfile))
    return config, model, train_losses, test_losses, accuracies


def visualize_images():
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True)
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # Pick 30 samples randomly
    indices = torch.randperm(len(trainset))[:30]
    images = [np.asarray(trainset[i][0]) for i in indices]
    labels = [trainset[i][1] for i in indices]
    # Visualize the images using matplotlib
    fig = plt.figure(figsize=(10, 10))
    for i in range(30):
        ax = fig.add_subplot(6, 5, i+1, xticks=[], yticks=[])
        ax.imshow(images[i])
        ax.set_title(classes[labels[i]])


@torch.no_grad()
def visualize_attention(model, output=None, device="cuda"):
    """
    Visualize the attention maps of the first 4 images.
    """
    model.eval()
    # Load random images
    num_images = 30
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # Pick 30 samples randomly
    indices = torch.randperm(len(testset))[:num_images]
    raw_images = [np.asarray(testset[i][0]) for i in indices]
    labels = [testset[i][1] for i in indices]
    # Convert the images to tensors
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    images = torch.stack([test_transform(image) for image in raw_images])
    # Move the images to the device
    images = images.to(device)
    model = model.to(device)
    # Get the attention maps from the last block
    logits, attention_maps = model(images, output_attentions=True)
    # Get the predictions
    predictions = torch.argmax(logits, dim=1)
    # Concatenate the attention maps from all blocks
    attention_maps = torch.cat(attention_maps, dim=1)
    # select only the attention maps of the CLS token
    attention_maps = attention_maps[:, :, 0, 1:]
    # Then average the attention maps of the CLS token over all the heads
    attention_maps = attention_maps.mean(dim=1)
    # Reshape the attention maps to a square
    num_patches = attention_maps.size(-1)
    size = int(math.sqrt(num_patches))
    attention_maps = attention_maps.view(-1, size, size)
    # Resize the map to the size of the image
    attention_maps = attention_maps.unsqueeze(1)
    attention_maps = F.interpolate(attention_maps, size=(32, 32), mode='bilinear', align_corners=False)
    attention_maps = attention_maps.squeeze(1)
    # Plot the images and the attention maps
    fig = plt.figure(figsize=(20, 10))
    mask = np.concatenate([np.ones((32, 32)), np.zeros((32, 32))], axis=1)
    for i in range(num_images):
        ax = fig.add_subplot(6, 5, i+1, xticks=[], yticks=[])
        img = np.concatenate((raw_images[i], raw_images[i]), axis=1)
        ax.imshow(img)
        # Mask out the attention map of the left image
        extended_attention_map = np.concatenate((np.zeros((32, 32)), attention_maps[i].cpu()), axis=1)
        extended_attention_map = np.ma.masked_where(mask==1, extended_attention_map)
        ax.imshow(extended_attention_map, alpha=0.5, cmap='jet')
        # Show the ground truth and the prediction
        gt = classes[labels[i]]
        pred = classes[predictions[i]]
        ax.set_title(f"gt: {gt} / pred: {pred}", color=("green" if gt==pred else "red"))
    if output is not None:
        plt.savefig(output)
    plt.show()


def pad_at_dim(t, pad: Tuple[int, int], *, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def slice_at_dim(t, dim_slice: slice, *, dim):
    dim += (t.ndim if dim < 0 else 0)
    colons = [slice(None)] * t.ndim
    colons[dim] = dim_slice
    return t[tuple(colons)]

def pad_or_slice_to(t, length, *, dim, pad_value = 0):
    curr_length = t.shape[dim]

    if curr_length < length:
        t = pad_at_dim(t, (0, length - curr_length), dim = dim, value = pad_value)
    elif curr_length > length:
        t = slice_at_dim(t, slice(0, length), dim = dim)

    return t


# helper functions
def exists(v):
    return v is not None

@contextmanager
def combine_contexts(a, b):
    with a() as c1, b() as c2:
        yield (c1, c2)

def find_first(cond: Callable, arr):
    for el in arr:
        if cond(el):
            return el

    return None


# adds a context manager for wandb tracking with a specific project and experiment name
def add_wandb_tracker_contextmanager(
    accelerator_instance_name = 'accelerator',
    tracker_hps_instance_name = 'tracker_hps'
):
    def decorator(klass):

        @contextmanager
        def wandb_tracking(
            self,
            project: str,
            run: Optional[str] = None,
            hps: Optional[dict] = None
        ):
            maybe_accelerator = getattr(self, accelerator_instance_name, None)

            assert exists(maybe_accelerator) and isinstance(maybe_accelerator, Accelerator), f'Accelerator instance not found at self.{accelerator_instance_name}'

            hps = getattr(self, tracker_hps_instance_name, hps)

            maybe_accelerator.init_trackers(project, config = hps)

            wandb_tracker = find_first(lambda el: isinstance(el, WandBTracker), maybe_accelerator.trackers)

            assert exists(wandb_tracker), 'wandb tracking was not enabled. you need to set `log_with = "wandb"` on your accelerate kwargs'

            if exists(run):
                assert exists(wandb_tracker)
                wandb_tracker.run.name = run

            yield

            maybe_accelerator.end_training() 

        if not hasattr(klass, 'wandb_tracking'):
            klass.wandb_tracking = wandb_tracking

        return klass

    return decorator


# automatically unwrap model when attribute cannot be found on the maybe ddp wrapped main model
class ForwardingWrapper:
  def __init__(self, parent, child):
    self.parent = parent
    self.child = child

  def __getattr__(self, key):
    if hasattr(self.parent, key):
      return getattr(self.parent, key)

    return getattr(self.child, key)

  def __call__(self, *args, **kwargs):
    call_fn = self.__getattr__('__call__')
    return call_fn(*args, **kwargs)

def auto_unwrap_model(
    accelerator_instance_name = 'accelerator',
    model_instance_name = 'model'
):
    def decorator(klass):
        _orig_init = klass.__init__

        def __init__(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            model = getattr(self, model_instance_name)
            accelerator = getattr(self, accelerator_instance_name)

            assert isinstance(accelerator, Accelerator)
            forward_wrapped_model = ForwardingWrapper(model, accelerator.unwrap_model(model))
            setattr(self, model_instance_name, forward_wrapped_model)

        klass.__init__ = __init__
        return klass

    return decorator


# gradient accumulation context manager
# for no_sync context on all but the last iteration
def model_forward_contexts(
    accelerator: Accelerator,
    model: Module,
    grad_accum_steps: int = 1
):
    for i in range(grad_accum_steps):
        is_last_step = i == grad_accum_steps - 1

        maybe_no_sync = partial(accelerator.no_sync, model) if not is_last_step else nullcontext

        yield partial(combine_contexts, accelerator.autocast, maybe_no_sync)