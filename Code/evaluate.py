import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.metrics import multiclass_dice_coeff, BoundaryLoss, hausdorff_distance


@torch.inference_mode()
def evaluate(net, dataloader, device, criterion, is_tversky=False, is_jaccard=False, is_boundary=False, alpha=0.5, beta=0.5):
    """
    Evaluates the performance of the model on a given dataset.
    This function computes the loss and evaluation metric for the model on the given dataset,
    allowing for the use of different loss functions and scores, depending on the specified flags.

    :param net: (torch.nn.Module): The neural network model to be evaluated.
    :param dataloader: (torch.utils.data.DataLoader): The DataLoader providing the evaluation dataset.
    :param device: (torch.device): The device (CPU or GPU) on which to perform the evaluation.
    :param criterion: (torch.nn.Module): Dice/Tversky loss function
    :param is_tversky: (bool, optional): If True, use the Tversky loss for evaluation. Default is False.
    :param is_jaccard: (bool, optional): If True, use the Jaccard score for evaluation. Default is False.
    :param is_boundary: (bool, optional): If True, add boundary regularization. Default is False
    :param alpha: (float, optional): Alpha parameter for the Tversky loss, balancing false positives and false negatives.
                Default is 0.5.
    :param beta: (float, optional): Beta parameter for the Tversky loss, balancing false positives and false negatives.
                Default is 0.5.
    :return:
        float: The aggregated loss over the dataset.
        float: The aggregated evaluation metric score (e.g., Dice or Jaccard) over the dataset.

    """

    net.eval()
    num_val_batches = len(dataloader)
    dice_score_val = 0
    epoch_val_loss = 0
    hausdorff_score_val = 0
    reg = 0.01
    counter_val = 0
    boundary_loss = BoundaryLoss(idc=[1])

    # 1. iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true, dist_map_label = batch['image'], batch['mask'], batch['distmap']

        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        mask_true = mask_true.to(device=device, dtype=torch.long)

        # predict the mask
        mask_pred = net(image)

        assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes]'
        dc_loss = criterion(
            F.softmax(mask_pred, dim=1).float(),
            F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float(),
            multiclass=True,
            is_tversky=is_tversky,
            alpha=alpha, beta=beta
        )
        if is_boundary:
            bl_loss = boundary_loss(F.softmax(mask_pred, dim=1), dist_map_label.to(device))
            val_loss = dc_loss + reg * bl_loss
        else:
            val_loss = dc_loss
        # convert to one-hot format
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
        mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
        # compute the Dice score, ignoring background
        dice_score_val += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], is_tversky=is_tversky,
                                                is_jaccard=is_jaccard, reduce_batch_first=False)
        HD_score, counter_val_batch = hausdorff_distance(mask_pred[:, 1:], mask_true[:, 1:])

        counter_val += counter_val_batch
        hausdorff_score_val += HD_score

        epoch_val_loss += val_loss.item()

    net.train()
    return dice_score_val, epoch_val_loss, hausdorff_score_val, counter_val
