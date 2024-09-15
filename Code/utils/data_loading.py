import logging
import numpy as np
import torch
from torch import Tensor
from PIL import Image
from functools import partial
from multiprocessing import Pool
import os
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.utils import depth, class2one_hot, one_hot2dist
from typing import Tuple, Callable, Union
from torchvision import transforms
from operator import itemgetter

D = Union[Image.Image, np.ndarray, Tensor]

def gt_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
    """
    Create a ground truth transformation pipeline that processes an image and converts it into a one-hot encoded tensor.
    :param resolution: (Tuple[float, ...]): The desired resolution for the transformation.
    :param K: (int): The number of classes for one-hot encoding.
    :return: Callable[[D], Tensor]: A callable function (transformation pipeline) that takes an image and returns a one-hot encoded PyTorch tensor.
    """
    return transforms.Compose([
        lambda img: np.array(img)[...],
        lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
        partial(class2one_hot, K=K),
        itemgetter(0)  # Then pop the element to go back to img shape
    ])

def dist_map_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
    """
    Create a transformation pipeline that converts ground truth images into distance maps.
    :param resolution: resolution: (Tuple[float, ...]): The desired resolution for the transformation.
    :param K: K: (int): The number of classes for one-hot encoding.
    :return: Callable[[D], Tensor]: A callable function (transformation pipeline) that takes an image,
     converts it into a one-hot encoded tensor, and then transforms it into a distance map, returning a PyTorch tensor.
    """
    return transforms.Compose([
        gt_transform(resolution, K),
        lambda t: t.cpu().numpy(),
        partial(one_hot2dist, resolution=resolution),
        lambda nd: torch.tensor(nd, dtype=torch.float32)
    ])

def load_image(filename, has_batches=False):
    """
    Load an image from a file or from a numpy array, depending on the input and the batch flag.
    :param filename: (str or np.ndarray): The path to the image file or a numpy array containing image data
    :param has_batches: (bool, optional): Flag indicating whether the dataset is organized in batches. Defaults to False.
    :return: Image: A PIL Image object loaded from the given filename or numpy array.
    """
    if has_batches:
        ext = isinstance(filename, np.ndarray)
        if ext:
            return Image.fromarray(filename)
        else:
            return Image.open(filename)
    else:
        ext = splitext(filename)[1]
        if ext:
            return Image.fromarray(np.load(filename))  # - original
        else:
            return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix, has_batches):
    """
    A function that extract the unique values of a mask
    :param idx: (int) The index of the image
    :param mask_dir: (str) Path to the directory containing the ground truth masks
    :param mask_suffix: (str, optional): Suffix to add to the image filenames to get the corresponding mask filenames.
    :param has_batches: (bool, optional): Flag indicating whether the dataset is organized in batches. Defaults to False.
    :return: (ndarray) THe unique values of the mask
    """
    if has_batches:
        # mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
        mask_file = list(mask_dir.glob(idx + mask_suffix + '.npy'))[0]
        # mask = np.asarray(load_image(mask_file))

        np_mask_file = np.load(mask_file)
        mask_list = []
        for idx1 in range(np_mask_file.shape[0]):
            mask = load_image(np_mask_file[idx1],
                              has_batches)
            if len(mask.size) == 2:
                mask_list.append(np.unique(mask))
            else:
                raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')
    else:
        mask_file = list(mask_dir.glob(idx.split("_")[0] + mask_suffix + "_" + idx.split("_")[-1] + '*'))[0]
        mask = np.asarray(load_image(mask_file, has_batches))
        if mask.ndim == 2:
            return np.unique(mask)
        elif mask.ndim == 3:
            mask = mask.reshape(-1, mask.shape[-1])
            return np.unique(mask, axis=0)
        else:
            raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')
    return mask_list


class BasicDataset(Dataset):
    """
    A dataset class for loading images and their corresponding masks, with several functionalities.
    This class is designed to be used with PyTorch's DataLoader to facilitate loading and preprocessing of images
    and masks for training, validating and evaluating.
    """
    def __init__(self, images_dir: str,  mask_dir: str, scale: float = 1.0, mask_suffix: str = '',
                 has_batches: bool = False, images_dir_plot: str = '', plot_flag: bool = False):
        """
        Initialize the BasicDataset class
        :param images_dir: (str): Path to the directory containing the original images.
        :param mask_dir: (str): Path to the directory containing the ground truth masks.
        :param scale: (float, optional) Value to scale the images with. Must be between 0 and 1. Defaults to 1.0.
        :param mask_suffix: (str, optional): Suffix to add to the image filenames to get the corresponding mask filenames. Defaults to ''.
        :param has_batches: (bool, optional): Flag indicating whether the dataset is organized in batches. Defaults to False.
        :param images_dir_plot: (str, optional): Path to the directory containing images for plotting, used if plot_flag is True. Defaults to ''.
        :param plot_flag: (bool, optional): Flag indicating whether to enable plotting functionality. Defaults to False.
        """
        self.images_dir = Path(images_dir)
        self.plot_flag = plot_flag
        if self.plot_flag:
            self.images_dir_plot = Path(images_dir_plot)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.has_batches = has_batches
        self.disttransform = dist_map_transform([1,1], 2)

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if
                    isfile(join(images_dir, file)) and not file.startswith('.') and file.endswith('.npy')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')

        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix,
                               has_batches=self.has_batches), self.ids), total=len(self.ids)
            ))
        if self.has_batches:
            all_unique = [list for lists in unique for list in lists]
            self.mask_values = list(sorted(np.unique(np.concatenate(all_unique), axis=0).tolist()))
        else:
            self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def choose_cases(self):
        """
        a function which creates separated files for each image - splitting the batch files
        Saves each image/mask in a new numpy file, instead of the original batched file
        """
        ids = self.ids.copy()

        ## for "Check the ratio between the three images dimenstions" section
        count_256 = 0
        count_320 = 0
        count_512 = 0
        # iterate over all the cases
        for case in ids:
            if self.has_batches:
                img_file = list(self.images_dir.glob(case + '.npy'))[0]
                if self.plot_flag:
                    img_file_plot = list(self.images_dir_plot.glob(case + '.npy'))[0]
                mask_file = list(self.mask_dir.glob(case + self.mask_suffix + '.npy'))[0]
            else:
                img_file = list(self.images_dir.glob(case.split("_")[0] + "_" + case.split("_")[-1] + '.npy'))[0]
                if self.plot_flag:
                    img_file_plot = list(self.images_dir_plot.glob(case.split("_")[0] + "_" + case.split("_")[-1] + '.npy'))[0]
                mask_file = list(self.mask_dir.glob(case.split("_")[0] + self.mask_suffix + "_" + case.split("_")[-1] + '.npy'))[0]

            np_img_file = np.load(img_file)
            if self.plot_flag:
                np_img_file_plot = np.load(img_file_plot)
            np_mask_file = np.load(mask_file)
            dim = np_img_file.shape[-1]
            ## Check the ratio between the three images dimenstions

            if dim == 256:
                count_256 += 1
            elif dim == 320:
                count_320 += 1
            else:
                count_512 += 1

            ## Choose only 512X512 images
            # if dim != 512:
            #     if os.path.exists(img_file):
            #         os.remove(img_file)
            #         # remove the case from the list of cases
            #         self.ids.remove(case)
            #     if os.path.exists(img_file_plot):
            #         os.remove(img_file_plot)
            #     if os.path.exists(mask_file):
            #         os.remove(mask_file)
            # resize the images to be from size 256X256

            ## Resize all images to 256X256
            if dim != 256:
                base_width = 256
                for i,item in enumerate(np_img_file):
                    PIL_image = Image.fromarray(np_img_file[i])
                    wprecent = base_width / float(PIL_image.size[0])
                    hsize = int((float(PIL_image.size[1])*float(wprecent)))
                    PIL_image = PIL_image.resize((base_width, hsize), Image.Resampling.LANCZOS)
                    if not os.path.exists(f'..{str(img_file).split(".")[2]}_{i}.npy'):
                        np.save(f'..{str(img_file).split(".")[2]}_{i}.npy', np.array(PIL_image))
                for i, item in enumerate(np_mask_file):
                    PIL_mask = Image.fromarray(np_mask_file[i])
                    wprecent = base_width / float(PIL_mask.size[0])
                    hsize = int((float(PIL_mask.size[1]) * float(wprecent)))
                    PIL_mask = PIL_mask.resize((base_width, hsize), Image.Resampling.LANCZOS)
                    if not os.path.exists(f'..{str(mask_file).split(".")[2]}_{i}.npy'):
                        np.save(f'..{str(mask_file).split(".")[2]}_{i}.npy', np.array(PIL_mask))
                if self.plot_flag:
                    for i, item in enumerate(np_img_file_plot):
                        PIL_image = Image.fromarray(np_img_file_plot[i])
                        wprecent = base_width / float(PIL_image.size[0])
                        hsize = int((float(PIL_image.size[1]) * float(wprecent)))
                        PIL_image = PIL_image.resize((base_width, hsize), Image.Resampling.LANCZOS)
                        if not os.path.exists(f'..{str(img_file_plot).split(".")[2]}_{i}.npy'):
                            np.save(f'..{str(img_file_plot).split(".")[2]}_{i}.npy', np.array(PIL_image))


            else:
                # save each image in every case as an independent ndarray
                for i, item in enumerate(np_img_file):
                    if not os.path.exists(f'..{str(img_file).split(".")[2]}_{i}.npy'):
                        np.save(f'..{str(img_file).split(".")[2]}_{i}.npy', item)
                if self.plot_flag:
                    for i, item in enumerate(np_img_file_plot):
                        if not os.path.exists(f'..{str(img_file_plot).split(".")[2]}_{i}.npy'):
                            np.save(f'..{str(img_file_plot).split(".")[2]}_{i}.npy', item)
                for i, item in enumerate(np_mask_file):
                    if not os.path.exists(f'..{str(mask_file).split(".")[2]}_{i}.npy'):
                        np.save(f'..{str(mask_file).split(".")[2]}_{i}.npy', item)

            # Remove the original case file, after saving each image in the case as an independent ndarray
            if os.path.exists(img_file):
                os.remove(img_file)
            if self.plot_flag:
                if os.path.exists(img_file_plot):
                   os.remove(img_file_plot)
            if os.path.exists(mask_file):
                os.remove(mask_file)
        print("320 images number:", count_320)
        print("512 images number:", count_512)
        print("256 images number:", count_256)


    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        """
        preprocess for the image obtained, including resizing and normalization for original image, and resizing
        and creating a new mask for the ground truth mask
        :param mask_values: (list) values of the mask. Our case includes binary masks, so values can be '0', '1' or both
        :param pil_img: (Image) mask or original image
        :param scale: (float) value to scale the images with
        :param is_mask: (bool) flag that determines if the image provided is mask or not
        :return: (ndarray) the preprocessed image
        """
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        #pil_img = transform.resize(pil_img, (newW, newH))
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BILINEAR)

        #pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        """
        Retrieve an item or a batch of items from the object using the given index.
        This method is used by PyTorch's DataLoader to access individual samples, or batches, during iteration.
        :param idx: (int) the case index
        :return:
            image: (Tensor) the original image/s
            mask: (Tensor) the ground truth mask/s
            distmap: (Tensor) the distance map/s for calculating boundary loss
        """
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name.split("_")[0] + self.mask_suffix + "_" + name.split("_")[-1] + '.npy'))
        img_file = list(self.images_dir.glob(name.split("_")[0] + "_" + name.split("_")[-1] + '.npy'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])
        dist_map_tensor: Tensor = self.disttransform(mask)

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'distmap': torch.as_tensor(dist_map_tensor).long().contiguous()
        }
