import argparse
import logging
import os
import cv2

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import convert_to_npy
from pathlib import Path
from evaluate import evaluate
from utils.metrics import dice_loss


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    """
    Gives the predicted mask for a given input image using a trained neural network model.
    :param net: (torch.nn.Module): The trained neural network model used to make mask predictions.
    :param full_img: (PIL.Image): The input image on which the mask prediction will be made.
    :param device: (torch.device): The device on which the computation is performed ('cpu' or 'cuda').
    :param scale_factor: (float, optional, default=1): Scales the input image before making predictions.
            A scale_factor of 1 means no scaling.
    :param out_threshold: (float, optional, default=0.5):
            The threshold used to convert model output probabilities into binary predictions.
            Values above this threshold are considered '1', while values below are considered '0'.
    :return: Predicted mask : numpy.ndarray
            The mask is a binary map obtained from the model's output.
    """

    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images')
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--source_dir', type=str, default='../Data/test_data', help='Directory to the original test data folder')
    parser.add_argument('--npy_seg_dir', type=str, default='../Data/test_set_seg_npy',
                        help='Directory to the splitted segmentation masks folder, containing .npy files')
    parser.add_argument('--npy_img_dir', type=str, default='../Data/test_set_orig_npy',
                        help='Directory to the splitted MR images folder, containing .npy files')
    parser.add_argument('--npy_img_dir_plot', type=str, default='../Data/test_set_orig_npy_plot',
                       help='Directory to the splitted MR images folder, containing .npy files, for plotting')
    parser.add_argument('--has_batches', type=bool, default=False,
                        help='A flag which activate the functions that create npy files for each image/mask in the cases files. ACTIVATE ONLY ON THE FIRST RUNNING')
    parser.add_argument('--alpha', type=float, default=0.3, help='Alpha variable - for tversky loss')
    parser.add_argument('--beta', type=float, default=0.7, help='Beta variable - for tversky loss')
    return parser.parse_args()


def get_output_filenames(args):
    """
    Creates a path to the predicted mask created
    :param args: all the parsers variables
    :return: args.output, filenames of output images / specific path for predicted image
    """

    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def plot_img_and_masks(orig_img_plot, gt_mask, pred_mask):
    """
    Plots the ground truth mask and predicted mask over the original image.
    :param orig_img_plot: (ndarray): The original image on which the masks will be overlaid.
    :param gt_mask: (ndarray): The ground truth segmentation mask to be overlaid on the original image.
    :param pred_mask: (ndarray): The predicted segmentation mask to be overlaid on the original image.
    This function displays the original image with both the ground truth and predicted masks superimposed for visual comparison.
    """

    gt_mask_np = np.array(gt_mask)
    # Find contours
    gt_contours, _ = cv2.findContours(gt_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create an empty image to draw the contours
    gt_perimeter_image = np.zeros_like(gt_mask_np)
    # Draw the contours (perimeter) on the empty image
    cv2.drawContours(gt_perimeter_image, gt_contours, -1, 1, thickness=2)
    # Extract the red channel of the gt_perimeter_image

    pred_mask_np = np.array(pred_mask).astype(np.uint8)
    has_mask = pred_mask_np.max()
    print("has mask: ", gt_mask_np.max())
    print("predict mask: ", has_mask)
    # Find contours
    pred_contours, _ = cv2.findContours(pred_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create an empty image to draw the contours
    pred_perimeter_image = np.zeros_like(pred_mask_np)
    # Draw the contours (perimeter) on the empty image
    cv2.drawContours(pred_perimeter_image, pred_contours, -1, 1, thickness=2)

    # Convert the grayscale image to RGB
    rgb_orig_img = np.stack([np.array(orig_img_plot)] * 3, axis=-1)
    rgb_orig_img = (rgb_orig_img - np.min(rgb_orig_img)) / (np.max(rgb_orig_img) - np.min(rgb_orig_img)) * 255
    rgb_orig_img = rgb_orig_img.astype(np.uint8)

    # green channel updated for gt_mask values
    gt_perimeter_mask = gt_perimeter_image > 0
    rgb_orig_img[gt_perimeter_mask, 0] = 0
    rgb_orig_img[gt_perimeter_mask, 1] = gt_perimeter_image[gt_perimeter_mask] * 255
    rgb_orig_img[gt_perimeter_mask, 2] = 0

    # red channel updated for pred_mask values
    pred_perimeter_mask = pred_perimeter_image > 0
    rgb_orig_img[pred_perimeter_mask, 0] = pred_perimeter_image[pred_perimeter_mask] * 255
    rgb_orig_img[pred_perimeter_mask, 1] = 0
    rgb_orig_img[pred_perimeter_mask, 2] = 0

    # Define colors and labels for the marks
    colors_labels = {
        'Predicted': 'red',
        'Ground Truth': 'green'
    }

    # Create a figure and axis for the image
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(rgb_orig_img,cmap='gray')
    # ax.imshow(gt_mask, cmap='gray')

    ax.set_title(f'Test Example - Predicted Vs. Ground Truth')

    # Hide the axis for the image
    ax.axis('off')

    # Create a separate axis for the legend/labels
    # Add extra space to the right of the image for labels
    fig.subplots_adjust(right=0.75)
    legend_ax = fig.add_axes([0.75, 0.2, 0.15, 0.8])  # [left, bottom, width, height]

    # Turn off axis for the legend
    legend_ax.axis('off')

    # Add labels for each color
    for label, color in colors_labels.items():
        legend_ax.plot([], [], color=color, label=label, linestyle='-', linewidth=3)

    # Add a legend to the legend axis
    legend_ax.legend(loc='center left')

    plt.show()


if __name__ == '__main__':
    args = get_args()
    # 1. create npy files from the raw data and add it to a new folder
    original_test_dir = args.source_dir
    seg_dirs_npy = args.npy_seg_dir
    orig_dirs_npy = args.npy_img_dir
    orig_dirs_npy_plot = args.npy_img_dir_plot

    if not os.path.exists(seg_dirs_npy):
        os.makedirs(seg_dirs_npy)
    if not os.path.exists(orig_dirs_npy):
        os.makedirs(orig_dirs_npy)
    if not os.path.exists(orig_dirs_npy_plot):
        os.makedirs(orig_dirs_npy_plot)

    to_plot = True  # False
    # 2. convert batches of images into separate files
    has_batches = args.has_batches
    if has_batches:
        convert_to_npy(original_test_dir, seg_dirs_npy, orig_dirs_npy, orig_dirs_npy_plot, to_plot)
    
    args.input = orig_dirs_npy
    args.output = seg_dirs_npy
    # 3. load our pre-trained model
    args.model = 'checkpoints/checkpoint_epoch44_risize_tversky_alpha_0.3_jaccard_boundary_haus.pth'
    

    # 3. Define some variables
    args.scale = 1
    img_scale = 1 #0.5

    is_tversky = True  # True => Tversky Loss, False => Dice Loss
    is_jaccard = True  # True => Jaccard Score, False => Dice Score
    is_boundary = False  # True => add boundary regularization, False => else
    if is_tversky:
        alpha = args.alpha
        beta = args.beta
    else:
        alpha = 0.5
        beta = 0.5
    # 4. Create test dataset
    test_dataset = BasicDataset(Path(orig_dirs_npy), Path(seg_dirs_npy), img_scale, mask_suffix='_segmentation',
                                has_batches=has_batches, images_dir_plot=Path(orig_dirs_npy_plot), plot_flag=to_plot)

    # 5. Choose cases containing 512x512 images + saving images separately
    if has_batches:
        test_dataset.choose_cases()
        print("Created npy files for each image and segmentation file. Re-run the code for predicting")
        sys.exit()

    # 6. Create dataloader
    test_loader = DataLoader(test_dataset, batch_size=58, shuffle=True, drop_last=False)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 7. Prepare parameter for evaluation
    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    criterion = dice_loss
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    empty_test_masks = 0
    total_test_masks = len(os.listdir('../Data/test_set_seg_npy'))
    object_pixels_test = 0
    total_pixels_test = len(os.listdir('../Data/test_set_seg_npy')) * (256 ** 2)
    for file in os.listdir('../Data/test_set_seg_npy'):
        mask_path = os.path.join('../Data/test_set_seg_npy', file)
        mask = np.load(mask_path)
        if mask.sum() == 0:
            empty_test_masks += 1
        object_pixels_test += mask.sum()
    empty_total_test_ratio = empty_test_masks / total_test_masks
    object_total_pixels_test_ratio = object_pixels_test / total_pixels_test


    num_test_batches = len(test_loader)
    # 8. Conduct evaluation
    test_score, test_loss, hausdorff_score_test, counter_test = evaluate(net, test_loader, device, criterion, is_tversky=is_tversky, is_jaccard=is_jaccard, is_boundary=is_boundary, alpha=alpha, beta=beta)
    norm_test_score = test_score / num_test_batches
    norm_test_hausdorff_dist = hausdorff_score_test / counter_test
    print('Test score:', norm_test_score)
    print('Hausdorff Distance: ', norm_test_hausdorff_dist)

    # 9. Create lists for MR images and their corresponding segmentation masks for plotting
    # Change the variable 'num_imgs_for_plot' to the number of images you want to plot
    count = 0
    num_imgs_for_plot = 10
    chosen_indxs = []
    img_list = []
    mask_list = []
    for i, filename in enumerate(os.listdir(out_files)):
        if any(filename.split("_")[0] in file for file in img_list):
            continue
        chosen_mask_filename = os.path.join(out_files, filename)
        gt_mask = Image.fromarray(np.load(chosen_mask_filename))
        if gt_mask.getextrema()[-1] > 0:
            if count > num_imgs_for_plot:
                chosen_ind = i
                break
            count += 1
            img_list.append(filename)
            mask_list.append(chosen_mask_filename)
            chosen_indxs.append(i)

    # 10. Plot the predicted mask and the GT mask on the image
    for idx, img in enumerate(img_list):
        img_path = f"{img.split('_')[0]}_{img.split('_')[-1]}"
        chosen_img_path = os.path.join(in_files,img_path)
        orig_img = Image.fromarray(np.load(chosen_img_path))
        orig_img_np = np.load(chosen_img_path)
        if to_plot:
            chosen_img_plot_path = os.path.join(orig_dirs_npy_plot,img_path)
            orig_img_plot = Image.fromarray(np.load(chosen_img_plot_path))
            orig_img_plot_np = np.load(chosen_img_plot_path)
        mask_gt_chosen = Image.fromarray(np.load(mask_list[idx]))

        pred_mask = predict_img(net=net,
                           full_img=orig_img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)


        plot_img_and_masks(orig_img_plot_np, mask_gt_chosen, pred_mask)

