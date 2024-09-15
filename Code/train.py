import argparse
import logging
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset
from utils.metrics import multiclass_dice_coeff, dice_coeff, BoundaryLoss, dice_loss, hausdorff_distance
from utils.utils import save_plots, convert_to_npy, test_val_ratio_calc


def train_model(
        model,
        device,
        has_batches: bool = False,
        epochs: int = 10,
        batch_size: int = 5,
        learning_rate: float = 1e-5,
        val_percent: float = 0.15,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        gradient_clipping: float = 1.0,
        alpha: float = 0.3,
        beta: float = 0.7,
):
    """
    Trains the given model on the provided dataset, with options for various training parameters.
    :param model: (torch.nn.Module): The model to be trained.
    :param device: (torch.device): The device on which to perform the training (e.g., CPU or GPU).
    :param has_batches: (bool, optional): Whether the dataset is divided into batches (default is False).
    :param epochs: (int, optional): The number of training epochs (default is 10).
    :param batch_size: (int, optional): The number of samples per batch (default is 5).
    :param learning_rate: (learning_rate): The learning rate for the optimizer (default is 1e-5).
    :param val_percent: (float, optional): The percentage of the dataset to use for validation (default is 0.15).
    :param save_checkpoint: (bool, optional): Whether to save the model checkpoint after each epoch (default is True).
    :param img_scale: (float, optional): Scaling factor for resizing images (default is 0.5).
    :param amp: (bool, optional): Whether to use automatic mixed precision for faster training (default is False).
    :param gradient_clipping: (float, optional): The maximum gradient norm for gradient clipping (default is 1.0).
    :param alpha: (float, optional): Weight for the FN component in tversky loss function (default is 0.3).
    :param beta: (float, optional): Weight for the FP component in tversky loss function (default is 0.7).

    """
    # 1. Create dataset
    has_batches = has_batches
    dataset = BasicDataset(dir_img, dir_mask, img_scale, mask_suffix='_segmentation', has_batches=has_batches)

    # 2. Choose cases containing 512x512 images / resizing images + saving images separately
    if has_batches:
        dataset.choose_cases()
        print("Created npy files for each image and segmentation file. Re-run the code for training")
        sys.exit()

    # 3. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(5))

    # 4. optional - calculate test validation ratios (if test data directories exist)
    to_calc_test_val_ratio = False
    if to_calc_test_val_ratio:
        test_dir = '../Data/test_set_seg_npy'  # path to test segmentation npy directory
        empty_total_ratios_diff, object_total_pixels_ratios_diff = test_val_ratio_calc(test_dir, dataset, val_set)

    # 5. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 6. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = dice_loss
    global_step = 0

    # 7. Begin training
    train_losses, val_losses = [], []
    train_dices, val_dices = [], []
    train_hausdorff, val_hausdorff = [], []
    best_score = 0.0
    # 8. Define some variables
    is_tversky = True  # True => Tversky Loss, False => Dice Loss
    is_jaccard = True  # True => Jaccard Score, False => Dice Score
    is_boundary = True # True => add boundary regularization, False => else
    if is_tversky:
        alpha = alpha
        beta = beta
    else:
        alpha = 0.5
        beta = 0.5

    addition = fr"- $ \alpha $ = {alpha}, $ \beta $ = {beta}"
    data_loss = f'Tversky Loss' if is_tversky else f'Dice Loss'
    data_score = 'Jaccard Score' if is_jaccard else 'Dice Score'
    # 9. Iterating over epochs
    for epoch in range(1, epochs + 1):
        counter_train = 0
        model.train()
        epoch_train_loss = 0
        dice_score_train = 0
        hausdorff_score_train = 0
        reg = 0.01 # regularization factor for boundary loss
        boundary_loss = BoundaryLoss(idc=[1])
        for batch in train_loader:
            images, true_masks, dist_map_label = batch['image'], batch['mask'], batch['distmap']
            assert images.shape[1] == model.n_channels, \
                f'Network has been defined with {model.n_channels} input channels, ' \
                f'but loaded images have {images.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'

            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                mask_pred = model(images)
                # boundary loss
                if is_boundary:
                    bl_loss = boundary_loss(F.softmax(mask_pred, dim=1),dist_map_label.to(device))
                dc_loss = criterion(
                    F.softmax(mask_pred, dim=1).float(),
                    F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                    multiclass=True,
                    is_tversky=is_tversky,
                    alpha=alpha, beta=beta
                )
                if is_boundary:
                    train_loss = dc_loss + reg * bl_loss
                else:
                    train_loss = dc_loss

                # convert to one-hot format
                true_masks = F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), model.n_classes).permute(0, 3, 1, 2).float()
                HD_score, counter_train_batch = hausdorff_distance(mask_pred[:, 1:], true_masks[:, 1:])
                counter_train += counter_train_batch
                hausdorff_score_train += HD_score
                # compute the Dice score, ignoring background
                dice_score_train += multiclass_dice_coeff(mask_pred[:, 1:], true_masks[:, 1:],
                                                          is_tversky=is_tversky, is_jaccard=is_jaccard,
                                                          reduce_batch_first=False)

            if epoch > 1:
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(train_loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

            global_step += 1
            epoch_train_loss += train_loss.item()
            experiment.log({
                'train loss': train_loss.item(),
                'step': global_step,
                'epoch': epoch
            })

        # Evaluation round
        dice_score_val, epoch_val_loss, hausdorff_score_val, counter_val = evaluate(model, val_loader, device, criterion,
                                                  is_tversky=is_tversky, is_jaccard=is_jaccard, is_boundary=is_boundary, alpha=alpha, beta=beta)
        if epoch > 1:
            scheduler.step(dice_score_val)

        # logging.info('Validation Dice score: {}'.format(dice_score_val))
        try:
            experiment.log({
                'learning rate': optimizer.param_groups[0]['lr'],
                'validation Dice': dice_score_val,
                'images': wandb.Image(images[0].cpu()),
                'masks': {
                    'true': wandb.Image(true_masks[0].float().cpu()),
                    'pred': wandb.Image(mask_pred.argmax(dim=1)[0].float().cpu()),
                },
                'step': global_step,
                'epoch': epoch
                # **histograms
            })
        except:
            pass
        # calculate all the normed losses and scores
        norm_epoch_train_loss = epoch_train_loss / len(train_loader)
        norm_epoch_val_loss = epoch_val_loss / len(val_loader)
        norm_dice_train_score = dice_score_train / len(train_loader)
        norm_dice_val_score = dice_score_val / len(val_loader)
        norm_hausdorff_score_train = hausdorff_score_train / counter_train #len(train_loader)
        norm_hausdorff_score_val = hausdorff_score_val / counter_val #len(val_loader)
        print(f"train loss epoch no.{epoch}: {norm_epoch_train_loss:.4f}")
        print(f"train dice score epoch no.{epoch}: {norm_dice_train_score:.4f}")
        print(f"validation loss epoch no.{epoch}: {norm_epoch_val_loss:.4f}")
        print(f"validation dice score epoch no.{epoch}: {norm_dice_val_score:.4f}")
        print(f"train hausdorff score epoch no.{epoch}: {norm_hausdorff_score_train:.4f}")
        print(f"validation hausdorff score epoch no.{epoch}: {norm_hausdorff_score_val:.4f}")
        train_losses.append(norm_epoch_train_loss)
        train_dices.append(norm_dice_train_score.cpu().numpy())
        val_losses.append(norm_epoch_val_loss)
        val_dices.append(norm_dice_val_score.cpu().numpy())
        train_hausdorff.append(norm_hausdorff_score_train)
        val_hausdorff.append(norm_hausdorff_score_val)
        # save weights of the trained model
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            # save the best model according to dice score on the validation set
            if norm_dice_val_score > best_score: # save the model that reached the best val score
                torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
                logging.info(f'Checkpoint {epoch} saved!')
        if norm_dice_val_score > best_score:
            best_score = norm_dice_val_score
    # save graphs of losses and scores
    save_plots(train_dices, train_losses, val_dices, val_losses, architecture='UNET', data_score=data_score,
               data_loss=data_loss, addition=addition, best_score=best_score)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--source_dir', type=str, default='../Data/training_data', help='Directory to the original training data folder')
    parser.add_argument('--npy_seg_dir', type=str, default='../Data/training_set_seg_npy', help='Directory to the splitted segmentation masks folder, containing .npy files')
    parser.add_argument('--npy_img_dir', type=str, default='../Data/training_set_orig_npy', help='Directory to the splitted MR images folder, containing .npy files')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/', help='Directory to the checkpoints folder, containing the models saved during training')
    parser.add_argument('--has_batches', type=bool, default=False,
                        help='A flag which activate the functions that create npy files for each image/mask in the cases files. ACTIVATE ONLY ON THE FIRST RUNNING')
    parser.add_argument('--alpha', type=float, default=0.3, help='Alpha variable - for tversky loss')
    parser.add_argument('--beta', type=float, default=0.7, help='Beta variable - for tversky loss')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    args.val = 10.0
    args.batch_size = 10
    args.epochs = 51
    args.scale = 1

    # 1. create directories
    source_dir = args.source_dir
    seg_dirs_npy = args.npy_seg_dir
    orig_dirs_npy = args.npy_img_dir
    ckpt_path = args.ckpt_dir
    dir_img = Path(orig_dirs_npy)
    dir_mask = Path(seg_dirs_npy)
    dir_checkpoint = Path(ckpt_path)

    if not os.path.exists(dir_checkpoint):
            os.makedirs(dir_checkpoint, exist_ok=True)
    if not os.path.exists(dir_mask):
        os.makedirs(dir_mask)
    if not os.path.exists(dir_img):
        os.makedirs(dir_img)
    has_batches = args.has_batches

    # 2. convert batches of images into separate files
    if has_batches:
        convert_to_npy(source_dir, seg_dirs_npy, orig_dirs_npy)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    # 3. Create model structure
    # Change here to adapt to your data
    # n_channels=3 for RGB images, n_channels=1 for grayscale images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    # 4. train our segmentation model
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            has_batches=args.has_batches,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            alpha=args.alpha,
            beta=args.beta
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            has_batches=args.has_batches,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            alpha=args.alpha,
            beta=args.beta
        )
