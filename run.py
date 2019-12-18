#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
from math import ceil

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cat_dataset import CatDataset
from unet import UNet

"""******************************
*** Constants/hyperparameters ***
******************************"""

# For information on running evaluation see the bottom of this file
CAT_IMAGE_DIR = 'cat_data'  # Path to cat data containing subdirectories TRAIN_SUBDIR and TEST_SUBDIR
# Directories containing 'input' and 'mask' directories with inputs and masks
TRAIN_DIR = os.path.join(CAT_IMAGE_DIR, 'Train')
TEST_DIR = os.path.join(CAT_IMAGE_DIR, 'Test')
# Subdirectories in TRAIN_DIR and TEST_DIR which contain inputs/masks
INPUT_SUBDIR = 'input'
MASK_SUBDIR = 'mask'
EPOCHS = 50  # How many epochs to run for 1.1/1.2/1.4
BATCH_SIZE = 2  # Batch size for 1.1/1.2/1.4 and the cat image training part of 1.3
MSRA_BATCH_SIZE = 8  # Batch size for training on MSRA10K
LEARNING_RATE = 0.0001  # Learning rate for 1.1/1.2/1.4
MSRA_LEARNING_RATE = 0.0001  # Learning rate for training on MSRA10K for 1.3
TRANSFER_LEARNING_RATE = 0.0001  # Learning rate for transfer learning after training on MSRA10K
# Epochs on which to save a collage of comparison images of the results on the test data (and contour images if enabled)
# dict for O(1) lookups since we search it every epoch
REPORT_EPOCHS = {25: True, 50: True, 100: True, 150: True}
COLLAGE_COLS = 2  # How many images per column in collages of test images
MSRA_EPOCHS = 10  # How many epochs to train on MSRA10K for transfer learning in 1.3
TRANSFER_EPOCHS = 10  # How many epochs to train on cat dataset when doing transfer learning in 1.3
COLLAGE_NO_DICE = False  # If False then write the dice coefficient on top of each image in collage of test images

"""**********************
*** General functions ***
**********************"""


def get_cat_loader(path, batch_size=4, **kwargs):
    """
    Returns a DataLoader for cat images and masks using CatDataset
    :param path: path to a directory containing INPUT_SUBDIR and MASK_SUBDIR subdirectories
    :param batch_size: batch size
    :param kwargs: arguments passed to CatDataset
    :return: DataLoader of CatDataset
    """
    dataset = CatDataset(path, **kwargs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def sdc(pred, mask, dim=1):
    """
    Returns the Sørensen–Dice coefficient between prediction pred and mask mask
    :param dim: the dimension in pred which is size 2 and contains the predictions for categories "not cat", "cat"
    :param pred: predicted mask
    :param mask: ground truth mask
    :return: Sørensen–Dice coefficient between pred and mask
    """
    _, pred = torch.max(pred, dim=dim)
    correct = (mask == pred).sum().float().item()
    incorrect = (mask != pred).sum().float().item()
    return 2 * correct / (2 * correct + incorrect)


def comparison_image(image, mask, pred, dice=None):
    """
    Returns an ndarray which is a BGR image of a horizontal concatenation of image, mask, and pred without modification
    All inputs should be in CPU memory
    :param image: base image tensor
    :param mask: mask as a tensor with values 0 or 1
    :param pred: predicted mask as a tensor with values 0 or 1
    :param dice: dice coefficient to write on the image or None to not write a dice coefficient
                 no dice coefficient will be written regardless of value if COLLAGE_NO_DICE = True
    """
    out = np.hstack([image.clone().detach().numpy().astype(np.uint8),
                     cv2.cvtColor(mask.clone().detach().numpy().astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR),
                     cv2.cvtColor(pred.clone().detach().numpy().astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)])
    if dice is None or COLLAGE_NO_DICE:
        return out
    return cv2.putText(out, "Dice = {:.4f}".format(dice), (0, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 90, 250))


def generate_collage(images):
    """
    Returns concatenation of images with COLLAGE_COLS columns per row
    :param images: images to make a collage of
    :return: image of concatenation of images with COLLAGE_COLS columns per row
    """
    rows = [np.hstack(images[r * COLLAGE_COLS: (r + 1) * COLLAGE_COLS]) for r in
            range(ceil(len(images) / COLLAGE_COLS))]
    missing_imgs = len(images) % COLLAGE_COLS
    if missing_imgs == 0:
        return np.vstack(rows)
    padding = [rows[-1]]
    padding.extend([np.zeros_like(images[0]) for i in range(len(images) % COLLAGE_COLS)])
    rows[-1] = np.hstack(padding)
    return np.vstack(rows)


def mask_argmax(mask):
    """
    Returns argmax(mask, dim=1)
    :param mask: mask tensor
    :return: argmax(mask, dim=1)
    """
    _, argmax = torch.max(mask, dim=1)
    return argmax


def print_epoch(epoch, train_loss, train_acc, test_loss, test_acc, total_epochs=EPOCHS):
    """
    Print the epoch information
    """
    print("\nEpoch {}/{}:".format(epoch + 1, total_epochs))
    print("\tTraining loss: {}".format(train_loss),
          "Sørensen–Dice coefficient: {}".format(train_acc))
    print("\tTest loss: {}".format(test_loss),
          "Sørensen–Dice coefficient: {}".format(test_acc))


"""*********************
**** Loss functions ****
*********************"""


def cross_entropy_loss(pred, mask):
    """
    Cross entropy loss calculated in a differentiable way
    :param pred: predicted mask, b*w*h
    :param mask: true mask, b*2*w*h
    :return: cross-entropy loss
    """
    pred = torch.nn.functional.normalize(pred.clone())
    # Softmax
    softmax = torch.nn.Softmax(dim=1)(pred)
    # Take log of softmax for each pixel/batch where index in dimension 1 corresponds to the value of mask (0 or 1)
    # Indices are reshaped so they broadcast with mask to have the index in dim1 be the value of mask
    nll = -torch.log(softmax[np.arange(mask.shape[0]).reshape((-1, 1, 1)),
                             mask,
                             np.arange(mask.shape[1]).reshape((1, -1, 1)),
                             np.arange(mask.shape[2]).reshape((1, 1, -1))])
    return nll.mean()


def huber_loss(pred, mask, d=2):
    diff = mask - torch.nn.Softmax(dim=1)(pred)[:, 1]
    return ((d ** 2) * (torch.sqrt(1 + (diff / d) ** 2) - 1)).mean()


"""*****************************
***** Question 1.1/1.2/1.4 *****
*****************************"""


def train_cats(augment: bool = False, contours: bool = False, weights_file: str = 'q1.pth'):
    loader = get_cat_loader(os.path.join('cat_data', 'Train'), batch_size=BATCH_SIZE, augment=augment)
    test_loader = get_cat_loader(os.path.join('cat_data', 'Test'), batch_size=1)
    # model = UNet()
    model = torch.load('model_transfer_95.pt')
    if torch.cuda.is_available():
        model.cuda()
    # The paper uses SGD as the optimizer but Adam gave me better results
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    """**** Change this line to test different loss functions for 1.1 ***"""
    criterion = cross_entropy_loss
    for e in range(0, EPOCHS):
        train_loss = 0
        train_accuracy = 0
        test_loss = 0
        test_accuracy = 0
        collage_images = []
        # Training
        for images, masks, colors in tqdm(loader):
            # Training step
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            # Update loss/accuracy totals and save image if requested
            train_loss += loss.item()
            train_accuracy += sdc(preds, masks)
        else:
            # Testing
            with torch.no_grad():
                for images, masks, colors in test_loader:
                    preds = model(images)
                    test_loss += criterion(preds, masks).item()
                    test_accuracy += sdc(preds, masks)
                    if e in REPORT_EPOCHS or e == EPOCHS - 1:
                        accuracies = [sdc(preds[i], masks[i], dim=0) for i in range(preds.shape[0])]
                        mask = mask_argmax(preds).cpu()
                        c = colors.cpu()
                        m = masks.cpu()
                        if contours:
                            for i in range(images.shape[0]):
                                collage_images.append(
                                    draw_contours(c[i].detach().cpu().numpy(), mask[i].detach().cpu().numpy()))
                        else:
                            for i in range(images.shape[0]):
                                collage_images.append(comparison_image(c[i], m[i], mask[i], accuracies[i]))
            if e in REPORT_EPOCHS or e == EPOCHS - 1:
                cv2.imwrite('{}.png'.format(e), generate_collage(collage_images))
            print_epoch(e, train_loss / len(loader), train_accuracy / len(loader), test_loss / len(test_loader),
                        test_accuracy / len(test_loader))
        torch.save(model.state_dict(), weights_file)


"""*********************
***** Question 1.3 *****
*********************"""


def segment_transfer_data():
    """
    Segment the data in MSRA10K into training and test sets
    Step 1/3 for 1.3
    """
    np.random.seed(100)  # for reproducibility
    base_dir = 'MSRA10K'
    if not os.path.exists(base_dir):
        print("Directory {} doesn't exist".format(base_dir))
        return
    test_dir = os.path.join(base_dir, 'Test')
    test_input_dir = os.path.join(test_dir, 'input')
    test_mask_dir = os.path.join(test_dir, 'mask')
    train_dir = os.path.join(base_dir, 'Train')
    train_input_dir = os.path.join(train_dir, 'input')
    train_mask_dir = os.path.join(train_dir, 'mask')
    # Split data randomly into 70% train, 30% test
    # The [1] is how many subdirectories into working directory the base_dir is + 1
    images = [str(os.path.split(f)[1].split('.')[0]) for f in glob.glob(os.path.join(base_dir, '*.png'))]
    np.random.shuffle(images)
    split = int(np.round(len(images) * 0.7))
    train = images[:split]
    test = images[split:]
    # Make directories
    os.mkdir(test_dir)
    os.mkdir(test_mask_dir)
    os.mkdir(test_input_dir)
    os.mkdir(train_dir)
    os.mkdir(train_mask_dir)
    os.mkdir(train_input_dir)
    # The masks are .png and the corresponding images are .jpg with same name
    for i in test:
        os.rename(os.path.join(base_dir, i + '.png'), os.path.join(test_mask_dir, i + '.png'))
        os.rename(os.path.join(base_dir, i + '.jpg'), os.path.join(test_input_dir, i + '.jpg'))
    for i in train:
        os.rename(os.path.join(base_dir, i + '.png'), os.path.join(train_mask_dir, i + '.png'))
        os.rename(os.path.join(base_dir, i + '.jpg'), os.path.join(train_input_dir, i + '.jpg'))


def learn_msra():
    """
    Learn from MSRA10K dataset
    Step 2/3 of 1.3
    *** This uses ~5GB of memory (CatDataset could be changed to use lazy loading if this is a problem) ***
    """
    loader = get_cat_loader(os.path.join('MSRA10K', 'Train'), batch_size=MSRA_BATCH_SIZE, augment=False)
    test_loader = get_cat_loader(os.path.join('MSRA10K', 'Test'), batch_size=1, augment=False)
    model = UNet()
    if torch.cuda.is_available():
        model.cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=MSRA_LEARNING_RATE)
    criterion = cross_entropy_loss
    for e in range(0, MSRA_EPOCHS):
        train_loss = 0
        train_accuracy = 0
        test_loss = 0
        test_accuracy = 0
        # Training
        for images, masks, colors in tqdm(loader):
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_accuracy += sdc(preds, masks)
        else:
            # Get test loss/accuracy
            with torch.no_grad():
                for images, masks, colors in test_loader:
                    preds = model(images)
                    test_loss += criterion(preds, masks).item()
                    test_accuracy += sdc(preds, masks)
            print_epoch(e, train_loss / len(loader), train_accuracy / len(loader), test_loss / len(test_loader),
                        test_accuracy / len(test_loader), total_epochs=MSRA_EPOCHS)
    torch.save(model, 'model_msra.pth')
    # Save the weights in a more portable format to be uploaded with the report
    torch.save(model.state_dict(), 'msra_weights.pth')


def transfer_learn(weights_file='q1_3.pth'):
    """
    Do the transfer learning using MSRA trained model as a starting point
    Step 3/3 of 1.3
    """
    loader = get_cat_loader(TRAIN_DIR, batch_size=2, augment=False)
    test_loader = get_cat_loader(TEST_DIR, batch_size=1)
    # Load model trained on MSRA10K from file
    model = torch.load('model_msra.pth')
    # Randomize weights for the last layer
    # (we could randomize more layers but experimentally I found doing just the last layer gave best results)
    model.layer9 = torch.nn.Sequential(
        torch.nn.Conv2d(128, 64, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 64, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 2, 1),
    )
    if torch.cuda.is_available():
        model.cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=TRANSFER_LEARNING_RATE)
    criterion = cross_entropy_loss
    for e in range(0, TRANSFER_EPOCHS):
        train_loss = 0
        train_accuracy = 0
        test_loss = 0
        test_accuracy = 0
        collage_images = []
        for images, masks, colors in tqdm(loader):
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_accuracy += sdc(preds, masks)
        else:
            # Get test loss/accuracy
            with torch.no_grad():
                for images, masks, colors in test_loader:
                    preds = model(images)
                    test_loss += criterion(preds, masks).item()
                    test_accuracy += sdc(preds, masks)
                    if e in REPORT_EPOCHS or e == TRANSFER_EPOCHS - 1:
                        accuracies = [sdc(preds[i], masks[i], dim=0) for i in range(preds.shape[0])]
                        mask = mask_argmax(preds).cpu()
                        c = colors.cpu()
                        m = masks.cpu()
                        for i in range(images.shape[0]):
                            collage_images.append(comparison_image(c[i], m[i], mask[i], accuracies[i]))
            if e in REPORT_EPOCHS or e == TRANSFER_EPOCHS - 1:
                cv2.imwrite('{}_transfer.png'.format(e), generate_collage(collage_images))
            print_epoch(e, train_loss / len(loader), train_accuracy / len(loader), test_loss / len(test_loader),
                        test_accuracy / len(test_loader), total_epochs=TRANSFER_EPOCHS)
        torch.save(model.state_dict(), weights_file)


# Function for question 1.4 (call train_cats with contours=True to use)
def draw_contours(image, mask):
    """
    Returns image with the contours of mask drawn on it
    :param image: input image
    :param mask: mask to get contours from
    :return: image with contours of mask drawn on it
    """
    mask = mask.astype(np.uint8).squeeze()
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result = image.copy()
    for c in contours:
        cv2.drawContours(result, c, -1, (0, 255, 0), 2)
    return result


def eval_on_dir(image_dir, weights_file='q1.pth', out_dir='output'):
    # Load model from disk
    model = UNet()
    model.load_state_dict(torch.load(weights_file))
    cuda = torch.cuda.is_available()
    if cuda:
        model.cuda()
    # Run inferences
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for filename in os.listdir(image_dir):
        image = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(image_dir, filename)), cv2.COLOR_BGR2GRAY), (128, 128))
        tensor = torch.from_numpy(image.reshape(1, 1, image.shape[0], image.shape[1])).float()
        if cuda:
            tensor = tensor.cuda()
        result = mask_argmax(model(tensor).detach()).cpu().numpy()[0] * 255
        cv2.imwrite(os.path.join(out_dir, filename), result)


if __name__ == '__main__':
    # 1.1
    train_cats(augment=False, weights_file='q1_1.pth')
    # 1.2
    train_cats(augment=True, weights_file='q1_2.pth')
    # 1.3
    segment_transfer_data()
    learn_msra()
    transfer_learn()
    # 1.4
    train_cats(contours=True)
    """
    To do inference put all images in a directory and update the arguments below
    The first argument should be a directory containing only image files
    The outputs will be saved in out_dir with the same name as the inputs
    """
    # eval_on_dir('images', weights_file='q1.pth', out_dir='output')
