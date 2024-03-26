import torch
from torch import Tensor
from torch.utils.data import Sampler

from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Tuple


class ImageBatchSampler(Sampler[int]):

    def __init__(self, data_source: Sized, num_imgs: int, same_imgs_size: int, batch_size: int,
                 image_resolution: Tuple[int, int], generator=None) -> None:
        super().__init__(data_source)

        self.data_source = data_source
        self.num_imgs = num_imgs
        self.batch_size = batch_size
        self.same_imgs_size = same_imgs_size
        self.image_w, self.image_h = image_resolution

        assert batch_size % same_imgs_size == 0, "Batch size must be divisible by same_imgs_size"

        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        self.generator = generator
        self.device = self.generator.device

    def __iter__(self) -> Iterator[int]:
        available_mask = torch.ones((self.num_imgs, self.image_h, self.image_w), dtype=torch.bool, device=self.device)
        # Gen within-image batch size
        img_batch_size = self.batch_size // self.same_imgs_size

        # Get the count of available pixels in each image
        imgs_available_pixs = available_mask.sum(dim=(1, 2))  # num pixels available in each image
        imgs_batch_size_pixs = imgs_available_pixs >= img_batch_size  # images with at least img_batch_size pixels
        finished = False

        while not finished:
            img_idx = torch.multinomial(imgs_batch_size_pixs.float(),
                                        num_samples=self.same_imgs_size, generator=self.generator)
            img_mask = available_mask[img_idx]
            img_pixs_idxs = torch.multinomial(img_mask.reshape(self.same_imgs_size, -1).float(),
                                              num_samples=img_batch_size, generator=self.generator)

            unraveled_idxs = torch.cat([
                img_idx.view(-1, 1).repeat(1, img_batch_size).view(-1, 1),  # image index
                img_pixs_idxs.view(-1, 1) // self.image_w,  # pixel y (height)
                img_pixs_idxs.view(-1, 1) % self.image_w  # pixel x (width)
            ], dim=-1)
            raveled_idx = (unraveled_idxs[:, 0] * self.image_w * self.image_h +
                           unraveled_idxs[:, 1] * self.image_w +
                           unraveled_idxs[:, 2])
            yield raveled_idx.cpu().tolist()  # Yield globally raveled pixels

            assert available_mask[list(unraveled_idxs.T)].all(), "Pixel was already used"
            available_mask[list(unraveled_idxs.T)] = False  # Mark pixels as unavailable

            imgs_available_pixs = available_mask.sum(dim=(1, 2))  # num pixels available in each image
            imgs_batch_size_pixs = imgs_available_pixs >= img_batch_size  # images with at least img_batch_size pixels
            # We finish as soon as there are less than same_imgs_size images with at least img_batch_size pixels
            finished = imgs_batch_size_pixs.sum() < self.same_imgs_size
