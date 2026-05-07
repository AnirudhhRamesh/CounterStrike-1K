import matplotlib.pyplot as plt

import torch
import numpy as np
from pathlib import Path
import lpips
from PIL import Image

def denormalize(tensor):
    return tensor * 0.5 + 0.5 # converts from -1, 1 to 0, 1

def show_sample(dataset, idx):
    sample = dataset[idx]
    frame = sample["frame"]
    actions = sample["actions"]

    img = denormalize(frame)
    img = img.permute(1,2,0) #C, H, W to H, W, C
    img = img.numpy().clip(0,1) #Clamp floating point drifts

    plt.figure(figsize=(4,4))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"frame_{idx}")
    plt.show()

def show_batch_sample(batch, n):
    if n > len(batch['frame']):
        print('n is larger than batch size, using batch_size instead')
        n = len(batch['frame'])

    frames = batch['frame']
    actions = batch['actions']

    fig, axes = plt.subplots(1 ,n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]
    for i in range(n):
        img = frames[i]
        img = denormalize(img)
        img = img.permute(1,2,0)
        img = img.numpy().clip(0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f"frame_{i}")
        axes[i].axis('off')

        #TODO: Write a function that also displays actions?

    plt.tight_layout()
    # plt.show()

def show_sequence_sample(batch, num_timesteps, batch_idx=0):
    B, T, C, H, W = batch['video'].shape

    if num_timesteps > T:
        print('num_timesteps is larger than sequence_length, using sequence_length instead')
        num_timesteps = T

    frames = batch['video'][batch_idx]
    # actions = batch['actions']

    fig, axes = plt.subplots(1, num_timesteps, figsize=(4 * num_timesteps, 4))
    if num_timesteps == 1:
        axes = [axes]
    for t in range(num_timesteps):
        img = frames[t]
        img = denormalize(img)
        img = img.permute(1,2,0)
        img = img.detach().cpu().numpy().clip(0, 1)
        
        axes[t].imshow(img)
        axes[t].set_title(f"frame_{t}")
        axes[t].axis('off')

    plt.tight_layout()

def show_sequence_gif(batch, num_timesteps, filename, output_path, batch_idx=0):
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    B, T, C, H, W = batch['video'].shape

    if num_timesteps > T:
        print('num_timesteps is larger than sequence_length, using sequence_length instead')
        num_timesteps = T
    if num_timesteps <= 0:
        raise ValueError("num_timesteps must be >= 1")

    frames = batch['video'][batch_idx]
    # actions = batch['actions']

    images = []
    for t in range(num_timesteps):
        img = frames[t]
        img = denormalize(img)
        img = img.permute(1,2,0) #C, H, W -> H, W, C
        img = img.detach().cpu().numpy().clip(0, 1)
        img = (img * 255).astype(np.uint8)
        images.append(Image.fromarray(img))

    if not images:
        raise ValueError("No frames available to save as GIF")

    output_file = output_dir / filename
    if output_file.suffix.lower() != ".gif":
        output_file = output_file.with_suffix(".gif")

    images[0].save(
        output_file,
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0,
    )
    return output_file

def perceptual_lpips(x_hat, x, device):
    lpips_fn = lpips.LPIPS(net="vgg").eval().to(device)
    
    for p in lpips_fn.parameters():
        p.requires_grad_(False)
    # LPIPS expects [-1,1] float tensors, NCHW
    return lpips_fn(x_hat, x).mean()

def save_ckpt(path, model, optimizer=None, epoch=None, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model": model.state_dict(),
        "epoch": epoch,
        "extra": extra
    }
    
    if hasattr(model, 'beta') and hasattr(model, 'lam_perc'):
        ckpt["config"]: {"beta": model.beta, "lam_perc": model.lam_perc}

    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()

    torch.save(ckpt, path)

def load_ckpt(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)

    model.load_state_dict(ckpt["model"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    start_epoch = (ckpt.get("epoch") or 0) + 1
    return ckpt, start_epoch