#!/usr/bin/env python3
import torch
import fire

def print_param_size(checkpoint: str, map_location: str = "cpu"):
    """
    Load a PyTorch model or state_dict checkpoint and print:
      - total number of parameters
      - number of trainable parameters
      - total size in bytes
    """
    ckpt = torch.load(checkpoint, map_location=map_location)

    # if it's a full model object
    if hasattr(ckpt, "parameters"):
        params = list(ckpt.parameters())
        total = sum(p.numel() for p in params)
        trainable = sum(p.numel() for p in params if p.requires_grad)
        size_bytes = sum(p.numel() * p.element_size() for p in params)

    # if it's a state_dict (or dict with "state_dict" key)
    elif isinstance(ckpt, dict):
        state_dict = ckpt.get("state_dict", ckpt)
        tensors = state_dict.values()
        total = sum(v.numel() for v in tensors)
        trainable = total
        size_bytes = sum(v.numel() * v.element_size() for v in tensors)

    else:
        raise ValueError("Unsupported checkpoint format")

    print(f"Total parameters:     {total}")
    print(f"Trainable parameters: {trainable}")
    print(f"Total size (bytes):   {size_bytes}")

if __name__ == "__main__":
    fire.Fire(print_param_size)

