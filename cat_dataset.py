import os

import cv2
import numpy as np
import torch
from cv2 import imread, resize, cvtColor, COLOR_BGR2GRAY
from torch.utils.data import Dataset
from torchvision import transforms

# Total number of possible transformations for augmentation (including no transformation)
NUM_TRANSFORMS = 5


class CatDataset(Dataset):
    """
    Cat dataset with cat images and masks
    Stores the GRAY and BGR images as well as the masks (as binary ndarrays) in memory
    When augmentation is enabled, the augmentations are done lazily to save memory
    """

    def __init__(self, directory: str, dim: int = 128, augment: bool = False, max_scale: float = 1.2):
        """
        :param directory: path to directory containing input and mask subdirectories
        :param dim: images are resized to dim*dim
        :param augment: whether to augment the data with random transformations (see self.__augment)
        :param max_scale: max scale multiplier when augmenting
        """
        self.dim = dim
        input_dir = os.path.join(directory, 'input')
        mask_dir = os.path.join(directory, 'mask')
        self.augment = augment
        self.max_scale = max_scale
        self.color_images = []
        # Store images as ndarrays so we can do transforms on them when augmenting
        self.inputs = [self.__load_image(os.path.join(input_dir, f), True) for f in os.listdir(input_dir)]
        self.masks = [self.__load_image(os.path.join(mask_dir, f)) for f in os.listdir(mask_dir)]
        self.masks = [self.__im2mask(i) for i in self.masks]
        assert len(self.inputs) == len(self.masks)
        self.to_tensor = transforms.ToTensor()

    def __load_image(self, filename, add_color=False):
        """
        Returns image at filename resized to self.dim*self.dim
        :param filename: image file
        :param add_color: if True then also append the colour version of the image to self.color_images
        :return: image resized to self.dim*self.dim
        """
        image = imread(filename)
        if image is None or image.data is None:
            raise FileNotFoundError("Invalid image file {}".format(filename))
        color = resize(image, (self.dim, self.dim))
        if add_color:
            self.color_images.append(color.astype(np.float32))
        return cvtColor(color, COLOR_BGR2GRAY).astype(np.float32)

    def __im2mask(self, image):
        """
        Convert mask image into a binary matrix
        :param image: image to convert
        :return: mask as a binary matrix
        """
        return np.clip(image.astype(np.int64), 0, 1).squeeze()

    def __transform(self, i, transform=-1, random=False):
        """
        Apply transformation to image i
        Valid transformations are Rotate (0), Intensity Shift (1), Flip (2), and Scale (3)
        :param i: index of image to transform in self.images
        :param transform: transformation number to apply. If <0 or >3 then no transformation is applied.
        :param random: if True then apply a random transformation
        :return: transformed input, mask, color_image
        """
        if random:
            transform = np.random.randint(NUM_TRANSFORMS)
        center = (self.dim / 2., self.dim / 2.)  # Center of an image
        shape = self.inputs[0].shape
        # Rotate
        if transform == 0:
            rot_matrix = cv2.getRotationMatrix2D(center, np.random.uniform(1, 360), 1)
            return cv2.warpAffine(self.inputs[i].astype(np.uint8), rot_matrix, shape).astype(np.float32), \
                   cv2.warpAffine(self.masks[i].astype(np.uint8), rot_matrix, shape).astype(np.float32), \
                   cv2.warpAffine(self.color_images[i].astype(np.uint8), rot_matrix, shape).astype(np.float32)
        # Intensity shift
        if transform == 1:
            # Do intensity shift in color then convert
            gray = cv2.cvtColor(np.clip(self.color_images[i] * (1 + np.random.uniform(-0.1, 0.1)), 0, 255),
                                COLOR_BGR2GRAY)
            return gray.astype(np.float32), self.masks[i].astype(np.float32), self.color_images[i].astype(np.float32)
        # Flip
        if transform == 2:
            flip = np.random.randint(-1, 2)  # 0 = flip on x, 1 = flip on y, -1 = flip on both
            return cv2.flip(self.inputs[i], flip).astype(np.float32), cv2.flip(self.masks[i], flip).astype(
                np.float32), cv2.flip(self.color_images[i], flip).astype(np.float32)
        # Scale
        if transform == 3:
            scale = np.random.uniform(1, self.max_scale)
            scaled_dim = round(scale * self.dim)
            scaled_input = cv2.resize(self.inputs[i], (scaled_dim, scaled_dim))
            scaled_mask = cv2.resize(self.masks[i], (scaled_dim, scaled_dim))
            scaled_color = cv2.resize(self.color_images[i], (scaled_dim, scaled_dim))
            # Crop
            start = np.random.randint(scaled_dim - self.dim)
            end = start + self.dim
            return scaled_input[start:end], scaled_mask[start:end], scaled_color[start:end]
        # No transform
        return self.inputs[i], self.masks[i], self.color_images[i]

    def __len__(self):
        if self.augment:
            return len(self.inputs) * NUM_TRANSFORMS
        return len(self.inputs)

    def __getitem__(self, i):
        if self.augment:
            transform = i // len(self.inputs)
            input, mask, color = self.__transform(i % len(self.inputs), transform=transform)
        else:
            input = self.inputs[i]
            mask = self.masks[i]
            color = self.color_images[i]
        color = color.astype(np.float32)
        if torch.cuda.is_available():
            return self.to_tensor(input).float().cuda(), self.to_tensor(mask).squeeze().long().cuda(), color
        return self.to_tensor(input).float(), self.to_tensor(mask).squeeze().long(), color
