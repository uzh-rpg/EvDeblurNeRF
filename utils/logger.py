import os
import torch
import wandb
import imageio
import numpy as np
from tensorboardX import SummaryWriter


class Logger(object):
    def __init__(self, log_dir, expname, use_wandb=False, use_tensorboard=True, wandb_id=None, args={}):
        """Create a summary writer logging to log_dir."""
        self.expname = expname

        self.use_wandb = use_wandb
        self.wandb = None
        if self.use_wandb:
            if wandb_id is not None:
                input("WARNING: Restoring wandb session with id {}. "
                      "Press ENTER to continue Ctrl-C to terminate".format(wandb_id))

            self.wandb = wandb.init(project="ev-deblur-nerf", name=expname, config=vars(args), id=wandb_id)
            self.wandb.config.update({'save_path': os.path.join(log_dir, expname)})
            self.wandb.log_code(".")
            assert wandb_id is None or self.wandb.id == wandb_id
            self.wandb_id = self.wandb.id

        self.use_tensorboard = use_tensorboard
        self.tensorboard = None
        if self.use_tensorboard:
            os.makedirs(os.path.join(log_dir, expname), exist_ok=True)
            self.tensorboard = SummaryWriter(os.path.join(log_dir, self.expname))
            print("Tensorboard logging to ", log_dir)

    def scalar(self, tag, value, step):
        """Log a scalar variable."""
        value = value.item() if isinstance(value, torch.Tensor) else value
        if self.tensorboard is not None:
            self.tensorboard.add_scalar(tag, value, step)
        if self.wandb is not None:
            self.wandb.log({tag: value}, step=step)

    def image(self, tag, images, step):
        """Log a list of images."""
        if self.tensorboard is not None:
            self.tensorboard.add_image(tag, images, step)
        if self.wandb is not None:
            self.wandb.log({tag: [wandb.Image(images, caption=tag)]}, step=step)

    def video(self, tag, path, value, step, fps=25, format="mp4"):
        shape = list(value.shape)

        # Make sure size is divisible by 2 (required by ffmpeg backend)
        shape[1] = int(np.ceil(shape[1] / 2) * 2)
        shape[2] = int(np.ceil(shape[2] / 2) * 2)
        value_pad = np.zeros(shape, dtype=value.dtype)
        value_pad[:, :value.shape[1], :value.shape[2]] = value

        imageio.mimwrite(path, value_pad, fps=fps, quality=8, macro_block_size=1)
        if self.wandb is not None:
            self.wandb.log({tag: wandb.Video(path, fps=fps, format=format)}, step=step)

    def histo(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        if self.tensorboard is not None:
            self.tensorboard.add_histogram(tag, values, step, bins=bins)
        if self.wandb is not None:
            self.wandb.log({tag: wandb.Histogram(values)}, step=step)