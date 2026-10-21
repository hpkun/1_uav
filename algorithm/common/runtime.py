import torch


def require_cuda(device='cuda'):
    requested = torch.device(device)
    if requested.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA is mandatory for training and evaluation; CPU fallback is disabled')
    return requested
