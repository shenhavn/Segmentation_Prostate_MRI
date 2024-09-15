import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import SimpleITK as sitk
import numpy as np
import torch
from torch import Tensor
from typing import cast, Set, Iterable, List, Tuple
from scipy.ndimage import distance_transform_edt as eucl_distance


def save_plots(train_score, train_loss, valid_score, valid_loss, architecture=None, data_score='Dice Score',
               data_loss="Dice Loss", addition="", best_score=0.0):
    """
    Saves the loss and accuracy plots to disk.
    :param train_score: (list): The normalized training score over epochs.
    :param train_loss: (list): The normalized training loss over epochs.
    :param valid_score: (list): The normalized validation score over epochs.
    :param valid_loss: (list): The normalized validation loss over epochs.
    :param architecture: (str, optional): The name of the model architecture (default is None).
    :param data_score: (str, optional): The label for the score metric (default is 'Dice Score').
    :param data_loss: (str, optional): The label for the loss metric (default is 'Dice Loss').
    :param addition: (str, optional): An additional string to append to the plot file names (default is an empty string).
    :param best_score: (float, optional): The best score achieved during training (default is 0.0).

    """

    tick_locations = list(np.arange(0, len(train_score), 5))
    tick_labels = list(np.arange(0, len(train_score), 5))
    # tick_labels[0] = 'Initialization'
    exp_indx = 0
    path_to_plot_score = os.path.join('plots',
                                      f'{architecture}_{data_score.split(" ")[0]}_{data_score.split(" ")[1]}_{exp_indx}.png')
    while os.path.exists(path_to_plot_score):
        exp_indx += 1
        path_to_plot_score = os.path.join('plots',
                                          f'{architecture}_{data_score.split(" ")[0]}_{data_score.split(" ")[1]}_{exp_indx}.png')
    exp_indx_2 = 0
    path_to_plot_loss = os.path.join('plots',
                                     f'{architecture}_{data_loss.split(" ")[0]}_{data_loss.split(" ")[1]}_{exp_indx_2}.png')
    while os.path.exists(path_to_plot_loss):
        exp_indx_2 += 1
        path_to_plot_loss = os.path.join('plots',
                                         f'{architecture}_{data_loss.split(" ")[0]}_{data_loss.split(" ")[1]}_{exp_indx_2}.png')

    # score plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_score, color='tab:blue', linestyle='-',
        label=f'Train {data_score}'
    )
    plt.plot(
        valid_score, color='tab:red', linestyle='-',
        label=f'Validation {data_score}'
    )
    plt.xticks(tick_locations, tick_labels)
    plt.xlabel('Epochs')
    plt.ylabel(f'{data_score}')
    plt.legend()
    plt.title(f"{data_score} Graphs {addition} - best score on valid set = {best_score:.4f}")

    plt.savefig(path_to_plot_score)
    plt.show()

    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='tab:blue', linestyle='-',
        label=f'Train {data_loss}'
    )
    plt.plot(
        valid_loss, color='tab:red', linestyle='-',
        label=f'Validation {data_loss}'
    )
    plt.xticks(tick_locations, tick_labels)
    plt.xlabel('Epochs')
    plt.ylabel(f'{data_loss}')
    plt.legend()
    plt.title(f"{data_loss} Graphs {addition}")

    plt.savefig(path_to_plot_loss)
    plt.show()


def convert_to_npy(original_dirs, seg_dirs_npy, orig_dirs_npy, orig_dirs_npy_plot='', plot_flag=False):
    """
    Converts all MR image and segmentation mask files to .npy format and organizes them into two folders:
    one for storing the MR image files and another for storing the segmentation mask files.
    :param original_dirs: (str): Directory to the path containing the original files (train/test) downloaded from the
            PROMISE12 website
    :param seg_dirs_npy: (str): Directory to the folder that will contain all segmentation mask files
    :param orig_dirs_npy: (str): Directory to the folder that will contain all MR images files
    :param orig_dirs_npy_plot: (str): Directory to the folder that will contain all MR images files for plotting
    :param plot_flag: (bool): flag which determines if to create 'orig_dirs_npy_plot' directory or not

    """

    for filename in os.listdir(original_dirs):
        if filename.lower().endswith('.raw') or filename.lower().endswith('.npy'):
            continue
        file_path = os.path.join(original_dirs, filename)
        # Convert SimpleITK image to numpy array
        sitk_image = sitk.ReadImage(file_path, outputPixelType=sitk.sitkUInt8)
        image_array = sitk.GetArrayFromImage(sitk_image)

        # Convert SimpleITK image to numpy array - for plotting directory
        if plot_flag:
            sitk_image_plot = sitk.ReadImage(file_path)
            image_array_plot = sitk.GetArrayFromImage(sitk_image_plot)

        # change seg_dirs to the path of the new filename
        if not os.path.exists(seg_dirs_npy):
            os.makedirs(seg_dirs_npy, exist_ok=True)

        if not os.path.exists(orig_dirs_npy):
            os.makedirs(orig_dirs_npy, exist_ok=True)
        # save to new folder containing only npy files
        if "segmentation" in filename:
            np.save(os.path.join(os.path.splitext(os.path.join(seg_dirs_npy, filename))[0] + '.npy'), image_array)
        else:
            np.save(os.path.join(os.path.splitext(os.path.join(orig_dirs_npy, filename))[0] + '.npy'), image_array)
            if plot_flag:
                np.save(os.path.join(os.path.splitext(os.path.join(orig_dirs_npy_plot, filename))[0] + '.npy'),
                        image_array_plot)


# Assert utils  -for implementing boundary loss
def uniq(a: Tensor) -> Set:
    """
    Convert a PyTorch tensor to a set of unique elements.
    :param a: (Tensor): Input tensor.
    :return: Set: A set containing the unique elements of the tensor.
    """

    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    """
    Check if the set of unique elements in a tensor is a subset of another iterable.
    :param a: (Tensor): Input tensor.
    :param sub: (Iterable): Iterable containing elements to check against.
    :return: bool: True if the set of unique elements in the tensor is a subset of the iterable, False otherwise.
    """

    return uniq(a).issubset(sub)


def simplex(t: Tensor, axis=1) -> bool:
    """
    Checks if a tensor is a simplex along a specified axis, meaning each slice along the axis sums to 1.
    :param t: (Tensor): The tensor to check. Each slice along the specified axis should sum to 1.
    :param axis: (int): The axis along which the tensor should be a simplex. Default is 1.
    :return: bool: True if the tensor is a simplex along the specified axis, False otherwise.
    """

    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    """
    Checks if a tensor is one-hot encoded along a specified axis.
    :param t: (Tensor): The tensor to check, expected to be one-hot encoded along the specified axis.
    :param axis: (int): The axis along which the tensor should be one-hot encoded. Default is 1.
    :return: bool: True if the tensor is one-hot encoded along the specified axis, False otherwise.
    """

    return simplex(t, axis) and sset(t, [0, 1])


def depth(e: List) -> int:
    """
    Compute the depth of nested lists
    """
    if type(e) == list and e:
        return 1 + depth(e[0])

    return 0


def class2one_hot(seg: Tensor, K: int) -> Tensor:
    """
    Converts a class-labeled segmentation tensor into a one-hot encoded tensor.
    :param seg: (Tensor): A tensor of shape (B, H, W, D) or (B, H, W) representing the segmentation masks,
                          where B is the batch size, and each element in the mask is a class label.
    :param K: (int): The number of classes.
    :return: Tensor: A one-hot encoded tensor of shape (B, K, H, W, D) or (B, K, H, W) where K is the number of classes.
    """

    # Breaking change but otherwise can't deal with both 2d and 3d
    # if len(seg.shape) == 3:  # Only w, h, d, used by the dataloader
    #     return class2one_hot(seg.unsqueeze(dim=0), K)[0]

    assert sset(seg, list(range(K))), (uniq(seg), K)

    b, *img_shape = seg.shape  # type: Tuple[int, ...]

    device = seg.device
    res = torch.zeros((b, K, *img_shape), dtype=torch.int32, device=device).scatter_(1, seg[:, None, ...], 1)

    assert res.shape == (b, K, *img_shape)
    assert one_hot(res)

    return res


def one_hot2dist(seg: np.ndarray, resolution: Tuple[float, float, float] = None,
                 dtype=None) -> np.ndarray:
    """
    Converts a one-hot encoded segmentation mask into a signed distance map.
    :param seg: (np.ndarray): One-hot encoded segmentation mask
    :param resolution: (Tuple[float, float, float], optional): Voxel size in each dimension. Default is None.
    :param dtype: (optional): Desired data type of the output array. Default is None, which preserves the dtype of seg.
    :return: np.ndarray: Signed distance map with the same shape as the input segmentation mask.
    """
    assert one_hot(torch.tensor(seg), axis=0)
    K: int = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        # Create a binary mask (positive mask)
        posmask = seg[k].astype(np.bool_)
        # If there are any positive pixels in the mask
        if posmask.any():
            negmask = ~posmask
            # Calculate the signed distance for each pixel
            res[k] = eucl_distance(negmask, sampling=resolution) * negmask \
                     - (eucl_distance(posmask, sampling=resolution) - 1) * posmask
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel

    return res


def test_val_ratio_calc(test_dir, dataset, val_set):
    '''
    Calculates difference between validation and test sets' ratios of empty masks and object pixels

    :param test_dir: string that contains the paths to validation and test directories
    :param dataset: BasicDataset object that contains the training set's images path before division to training & validation
    :param val_set: Subset object that contains the validation set's images path after division to training & validation
    :return:
        empty_total_ratios_diff: float that contains difference between validation and test sets' ratios of empty masks
        object_total_pixels_ratios_diff: float that contains difference between validation and test sets' ratios of object pixels

    '''

    empty_test_masks = 0
    total_test_masks = len(os.listdir(test_dir))
    object_pixels_test = 0
    total_pixels_test = len(os.listdir(test_dir)) * (512 ** 2)
    for file in os.listdir(test_dir):
        mask_path = os.path.join(test_dir, file)
        mask = np.load(mask_path)
        if mask.sum() == 0:
            empty_test_masks += 1
        object_pixels_test += mask.sum()
    empty_total_test_ratio = empty_test_masks / total_test_masks
    object_total_pixels_test_ratio = object_pixels_test / total_pixels_test

    empty_val_masks = 0
    total_val_masks = len(val_set)
    object_pixels_val = 0
    total_pixels_val = len(val_set) * (512 ** 2)
    for ind in val_set.indices:
        file = os.listdir(dataset.mask_dir)[ind]
        mask_path = os.path.join(dataset.mask_dir, file)
        mask = np.load(mask_path)
        if mask.sum() == 0:
            empty_val_masks += 1
        object_pixels_val += mask.sum()
    empty_total_val_ratio = empty_val_masks / total_val_masks
    object_total_pixels_val_ratio = object_pixels_val / total_pixels_val

    empty_total_ratios_diff = empty_total_test_ratio - empty_total_val_ratio
    object_total_pixels_ratios_diff = object_total_pixels_test_ratio - object_total_pixels_val_ratio

    return empty_total_ratios_diff, object_total_pixels_ratios_diff
