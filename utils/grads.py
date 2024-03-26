import torch
from torch.autograd import Function


def grads_norm(model, norm_type=2):
    results = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            try:
                results[name] = float(p.grad.data.norm(norm_type))
            except Exception:
                # this param had no grad
                pass

    total_norm = float(torch.tensor(list(results.values())).norm(norm_type))
    results['total'] = total_norm
    return results
