import torch
from torch import Tensor, einsum
from utils.utils import simplex, one_hot
from typing import List
import numpy as np
from skimage import metrics
from skimage.morphology import erosion



class SurfaceLoss():
    """
    A class to compute the boundary loss between predicted masks and ground truth masks
    Boundary loss is used to measure how well the predicted masks align with the ground truth masks in
    the distance maps.
    """
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        """
        Compute the boundary loss between predicted probabilities and distance maps.
        :param probs: (Tensor): The predicted mask tensor with shape (batch_size, num_classes, height, width).
        :param dist_maps: (Tensor): The distance maps tensor with shape (batch_size, num_classes, height, width).
        :return: (Tensor): The computed boundary loss.
        """
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bkwh,bkwh->bkwh", pc, dc)

        loss = multipled.mean()

        return loss


BoundaryLoss = SurfaceLoss

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    """
    Average of Dice coefficient for all batches, or for a single mask
    :param input: (Tensor): The predicted binary mask (tensor of shape [batch_size, ...])
    :param target: (Tensor): The ground truth binary mask (tensor of shape [batch_size, ...])
    :param reduce_batch_first: (bool): flag that tells if to reduce batch dimension or not
    :param epsilon: (float): A small value to avoid division by zero
    :return: dice.mean(): The mean Dice Coefficient as a tensor.
    """
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def tversky_coeff(input: Tensor, target: Tensor, epsilon: float = 1e-6, alpha: float = 0.5,
                  beta: float = 0.5) -> Tensor:
    """
    Computes the Tversky Coefficient between two tensors.
    :param input: (Tensor): The predicted binary mask (tensor of shape [batch_size, ...]).
    :param target: (Tensor): The ground truth binary mask (tensor of shape [batch_size, ...]).
    :param epsilon: (float): Weight for false positives.
    :param alpha: (float): Weight for false negatives.
    :param beta: (float): A small value to avoid division by zero.
    :return: tversky (Tensor): The Tversky Coefficient as a tensor.
    """

    # Ensure the input and target have the same shape
    assert input.size() == target.size(), "Input and target must have the same shape."

    # Flatten the tensors to calculate the intersection and difference across all dimensions
    input_flat = input.reshape(-1)
    target_flat = target.reshape(-1)

    # Compute the true positives (intersection)
    true_positives = (input_flat * target_flat).sum()

    # Compute the false positives and false negatives
    false_positives = ((1 - target_flat) * input_flat).sum()
    false_negatives = (target_flat * (1 - input_flat)).sum()

    # Compute the Tversky Coefficient
    tversky = (true_positives + epsilon) / (true_positives + alpha * false_positives + beta * false_negatives + epsilon)

    return tversky


def multiclass_dice_coeff(input: Tensor, target: Tensor, is_tversky: bool = False, is_jaccard: bool = False,
                        reduce_batch_first: bool = False, epsilon: float = 1e-6, alpha: float = 0.5,
                          beta: float = 0.5):
    """
    a function that given the flags "is_tersky" and "is_jaccard" calculates the Dice or Jaccard scores, and tversky
    coefficient for calculating tversky loss
    :param input: (Tensor): one-hot tensor of predicted masks
    :param target: (Tensor): one-hot tensor of ground truth masks
    :param is_tversky: (bool): True => Tversky Loss, False => Dice Loss
    :param is_jaccard: (bool): True => Jaccard Score, False => Dice Score
    :param reduce_batch_first: (bool): flag that tells if to reduce batch dimension or not
    :param epsilon: (float): A small value to avoid division by zero
    :param alpha: (float): Weight for false positives.
    :param beta: (float): Weight for false negatives.
    :return: (Tensor): Dice/jaccard/ tversky coefficient
    """
    # Average of Dice coefficient for all classes
    if is_tversky:
        return tversky_coeff(input.flatten(0, 1), target.flatten(0, 1), epsilon=epsilon, alpha=alpha, beta=beta)
    elif is_jaccard:
        return tversky_coeff(input.flatten(0, 1), target.flatten(0, 1), epsilon=epsilon, alpha=1, beta=1)
    else:  # dice
        return tversky_coeff(input.flatten(0, 1), target.flatten(0, 1), epsilon=epsilon, alpha=0.5, beta=0.5)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False, is_tversky: bool = False,
              is_jaccard: bool = False, alpha: float = 0.5, beta: float = 0.5):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, is_tversky=is_tversky, is_jaccard=is_jaccard, reduce_batch_first=True, alpha=alpha,
                  beta=beta)

def hausdorff_distance(pred_masks, GT_masks):
    """
    Calculate the Hausdorff distance between a "ground truth" and a "predicted" segmentation mask.
    :param pred_masks: (Tensor): Contains all the predicted masks in a batch
    :param GT_masks: (Tensor): Contains all the ground truth masks in a batch
    :return: distance: (float):
                The Hausdorff distance sum between all "ground truth" and all "predicted" segmentation mask,
                using the Euclidean distance.
            counter: int
                The number of masks that aren't empty and considered in the calculation of Hausdorff distance
    """
    # Creates "contours" image by xor-ing an erosion
    distances = 0
    pairs = []
    counter = 0
    for GT_mask, pred_mask in zip(GT_masks, pred_masks):
        se = np.array([[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]) # 3 dimensions in order to fit the masks size
        gt_contour = GT_mask.cpu().numpy().astype(int) ^ erosion(GT_mask.cpu().numpy().astype(int), se)
        predicted_contour = pred_mask.cpu().numpy().astype(int) ^ erosion(pred_mask.cpu().numpy().astype(int))
        # Computes & display the distance & the corresponding pair of points
        HD_score = metrics.hausdorff_distance(gt_contour, predicted_contour)
        if HD_score == np.inf:
            continue
        distances += metrics.hausdorff_distance(gt_contour, predicted_contour)
        pairs.append(metrics.hausdorff_pair(gt_contour, predicted_contour))
        counter += 1

    return distances, counter
