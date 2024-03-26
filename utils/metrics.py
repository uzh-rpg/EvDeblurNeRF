from skimage import metrics
import torch
import torch.hub
from networks.lpips import LPIPS
import numpy as np

img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

photometric = {
    "mse": None,
    "ssim": None,
    "psnr": None,
    "lpips": None
}


def compute_img_metric(im1t: torch.Tensor, im2t: torch.Tensor,
                       metric="mse", margin=0, mask=None, format=None):
    """
    im1t, im2t: torch.tensors with batched imaged shape, range from (0, 1)
    """
    if metric not in photometric.keys():
        raise RuntimeError(f"img_utils:: metric {metric} not recognized")
    if photometric[metric] is None:
        if metric == "mse":
            photometric[metric] = metrics.mean_squared_error
        elif metric == "ssim":
            photometric[metric] = metrics.structural_similarity
        elif metric == "psnr":
            photometric[metric] = metrics.peak_signal_noise_ratio
        elif metric == "lpips":
            photometric[metric] = LPIPS().cpu()

    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[1] == 1:
            mask = mask.expand(-1, 3, -1, -1)
        mask = mask.permute(0, 2, 3, 1).numpy()
        batchsz, hei, wid, _ = mask.shape
        if margin > 0:
            marginh = int(hei * margin) + 1
            marginw = int(wid * margin) + 1
            mask = mask[:, marginh:hei - marginh, marginw:wid - marginw]

    # convert from [0, 1] to [-1, 1]
    im1t = (im1t * 2 - 1).clamp(-1, 1)
    im2t = (im2t * 2 - 1).clamp(-1, 1)

    # Adds batch dimension if missing
    if (im1t.dim() == 3 and format is None) or format in ["HWC", "CHW"]:
        im1t = im1t.unsqueeze(0)
        im2t = im2t.unsqueeze(0)
    im1t = im1t.detach().cpu()
    im2t = im2t.detach().cpu()

    # Permute to (B, C, H, W) if in HWC
    if (im1t.shape[-1] == 3 and format is None) or format in ["BHWC", "HWC"]:
        im1t = im1t.permute(0, 3, 1, 2)
        im2t = im2t.permute(0, 3, 1, 2)

    # Reshape from (B, C, H, W) to (B, H, W, C)
    im1 = im1t.permute(0, 2, 3, 1).numpy()
    im2 = im2t.permute(0, 2, 3, 1).numpy()
    batchsz, hei, wid, _ = im1.shape
    if margin > 0:
        marginh = int(hei * margin) + 1
        marginw = int(wid * margin) + 1
        im1 = im1[:, marginh:hei - marginh, marginw:wid - marginw]
        im2 = im2[:, marginh:hei - marginh, marginw:wid - marginw]
    values = []

    for i in range(batchsz):
        if metric in ["mse", "psnr"]:
            if mask is not None:
                im1 = im1 * mask[i]
                im2 = im2 * mask[i]
            value = photometric[metric](
                im1[i], im2[i]
            )
            if mask is not None:
                hei, wid, _ = im1[i].shape
                pixelnum = mask[i, ..., 0].sum()
                value = value - 10 * np.log10(hei * wid / pixelnum)
        elif metric in ["ssim"]:
            value, ssimmap = photometric["ssim"](
                im1[i], im2[i], multichannel=True, full=True
            )
            if mask is not None:
                value = (ssimmap * mask[i]).sum() / mask[i].sum()
        elif metric in ["lpips"]:
            value = photometric[metric](
                im1t[i:i + 1], im2t[i:i + 1]
            )
        else:
            raise NotImplementedError
        values.append(value)

    return sum(values) / len(values)
