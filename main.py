from pyexpat import model
import torchvision.models as models
import torch.nn as nn
import torch
import os
import cv2
import PIL
import torchvision
import torchvision.transforms as transforms
import datetime
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from torchvision.utils import make_grid, save_image

import torchattacks

from adv_cam import GuidedBackpropReLUModel

import argparse


def normalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.sub(mean).div(std)


def denormalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.mul(std).add(mean)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return self.do(tensor)

    def do(self, tensor):
        return normalize(tensor, self.mean, self.std)

    def undo(self, tensor):
        return denormalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


def cam_to_adv(model='resnet50', img_dir='images',
               img_name='both.png', target_label=400,
               finalconv_name='layer1', active_layer=281,
               save_loc='./image_save_test', eps=0.05, steps=30
               ):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_lists = {'resnet18': models.resnet18,
                   'resnet34': models.resnet34,
                   'resnet50': models.resnet101,
                   'resnet152': models.resnet152}

    model = model_lists[model](pretrained=True).to(device)
    model.eval()
    img_path = os.path.join(img_dir, img_name)

    pil_img = PIL.Image.open(img_path)

    normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    torch_img = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
    torch_img = F.interpolate(torch_img, size=(224, 224), mode='bilinear', align_corners=False)  # (1, 3, 224, 224)
    normed_torch_img = normalizer(torch_img).to(device)  # (1, 3, 224, 224)

    atk = torchattacks.PGD(model, eps=eps, steps=steps)
    f = lambda x, y: y
    atk.set_mode_targeted_by_function(f)

    labels = torch.tensor([target_label])

    normed_torch_img = atk(normed_torch_img, labels)

    outputs = model(normed_torch_img)
    pred = torch.argmax(outputs, dim=1)
    print(f"prediction: {pred.item()}  target: {target_label}")

    img = np.transpose(torch_img.cpu().numpy().squeeze(), (1, 2, 0))  # (224, 224, 3)

    # activations
    feature_blobs = []

    # gradients
    backward_feature = []

    # output으로 나오는 feature를 feature_blobs에 append하도록
    def hook_feature(module, input, output):
        feature_blobs.append(output.cpu().data)

    # Grad-CAM
    def backward_hook(module, input, output):
        backward_feature.append(output[0])

    model._modules.get(finalconv_name).register_forward_hook(hook_feature)
    model._modules.get(finalconv_name).register_backward_hook(backward_hook)

    # get the softmax weight
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().detach().numpy())  # [1000, 512]

    # Prediction
    logit = model(normed_torch_img)
    h_x = F.softmax(logit, dim=1).data.squeeze()  # softmax 적용

    # ============================= #
    # ==== Grad-CAM main lines ==== #
    # ============================= #

    # Tabby Cat: 281, pug-dog: 254
    score = logit[:, active_layer].squeeze()  # 예측값 y^c
    score.backward(retain_graph=True)  # 예측값 y^c에 대해서 backprop 진행

    activations = feature_blobs[0].to(device)  # (1, 512, 7, 7), forward activations
    gradients = backward_feature[0]  # (1, 512, 7, 7), backward gradients
    b, k, u, v = gradients.size()

    alpha = gradients.view(b, k, -1).mean(2)  # (1, 512, 7*7) => (1, 512), feature map k의 'importance'
    weights = alpha.view(b, k, 1, 1)  # (1, 512, 1, 1)

    grad_cam_map = (weights * activations).sum(1, keepdim=True)  # alpha * A^k = (1, 512, 7, 7) => (1, 1, 7, 7)
    grad_cam_map = F.relu(grad_cam_map)  # Apply R e L U
    grad_cam_map = F.interpolate(grad_cam_map, size=(224, 224), mode='bilinear',
                                 align_corners=False)  # (1, 1, 224, 224)
    map_min, map_max = grad_cam_map.min(), grad_cam_map.max()
    grad_cam_map = (grad_cam_map - map_min).div(map_max - map_min).data  # (1, 1, 224, 224), min-max scaling

    # grad_cam_map.squeeze() : (224, 224)
    grad_heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam_map.squeeze().cpu()),
                                     cv2.COLORMAP_JET)  # (224, 224, 3), numpy

    grad_heatmap = np.float32(grad_heatmap) / 255

    grad_result = grad_heatmap + img
    grad_result = grad_result / np.max(grad_result)
    grad_result = np.uint8(255 * grad_result)

    # ============================= #
    # ==Guided-Backprop main lines= #
    # ============================= #

    # gb_model => ReLU function in resnet50 change to GuidedBackpropReLU.
    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
    gb_num = gb_model(torch_img, target_category=active_layer)
    gb = deprocess_image(gb_num)  # (224, 224, 3), numpy

    # Guided-Backpropagation * Grad-CAM => Guided Grad-CAM
    # See Fig. 2 in paper.
    # grad_cam_map : (1, 1, 224, 224) , torch.Tensor
    grayscale_cam = grad_cam_map.squeeze(0).cpu().numpy()  # (1, 224, 224), numpy
    grayscale_cam = grayscale_cam[0, :]  # (224, 224)
    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])  # (224, 224, 3)

    cam_gb = deprocess_image(cam_mask * gb_num)

    grad_heatmap = cv2.cvtColor(grad_heatmap, cv2.COLOR_BGR2RGB)
    grad_result = cv2.cvtColor(grad_result, cv2.COLOR_BGR2RGB)
    gb = cv2.cvtColor(gb, cv2.COLOR_BGR2RGB)
    cam_gb = cv2.cvtColor(cam_gb, cv2.COLOR_BGR2RGB)

    return grad_heatmap, grad_result, gb, cam_gb


def save_adv_cam(args):
    target_list = args.target_list
    row_nums = args.row_nums
    saved_loc = args.saved_loc

    os.makedirs(saved_loc, exist_ok=True)

    total_grad_heatmap = []
    total_grad_result = []
    total_gb = []
    total_cam_gb = []

    for target_label in target_list:
        grad_heatmap, grad_result, gb, cam_gb = cam_to_adv(
            model=args.model,
            img_dir=args.img_dir,
            img_name=args.img_name,
            target_label=target_label,
            finalconv_name=args.finalconv_name,
            active_layer=args.active_layer,
            save_loc=args.saved_loc,
            eps=args.eps,
            steps=args.steps
        )

        total_grad_heatmap.append(grad_heatmap)
        total_grad_result.append(grad_result)
        total_gb.append(gb)
        total_cam_gb.append(cam_gb)

    c1 = []  # total_grad_heatmap
    c2 = []  # total_grad_result
    c3 = []  # total_gb
    c4 = []  # total_cam_gb
    idx = 0

    columns = len(total_grad_heatmap) // row_nums

    for _ in range(columns):
        c1.append(cv2.hconcat(total_grad_heatmap[idx:idx + row_nums]))
        c2.append(cv2.hconcat(total_grad_result[idx:idx + row_nums]))
        c3.append(cv2.hconcat(total_gb[idx:idx + row_nums]))
        c4.append(cv2.hconcat(total_cam_gb[idx:idx + row_nums]))
        idx += row_nums

    # print(f"length of c1: {len(c1)}, c1[0]_shape: {c1[0].shape}, c1[1]_shape: {c1[1].shape}")

    c1 = cv2.vconcat(c1)
    c2 = cv2.vconcat(c2)
    c3 = cv2.vconcat(c3)
    c4 = cv2.vconcat(c4)

    plt.rcParams["figure.figsize"] = (5 * columns, 5 * row_nums)
    plt.imshow(c1)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.savefig(os.path.join(saved_loc, "grad_heatmap.jpg"))

    plt.rcParams["figure.figsize"] = (5 * columns, 5 * row_nums)
    plt.imshow(c2)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.savefig(os.path.join(saved_loc, "grad_result.jpg"))

    plt.rcParams["figure.figsize"] = (5 * columns, 5 * row_nums)
    plt.imshow(c3)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.savefig(os.path.join(saved_loc, "gb.jpg"))

    plt.rcParams["figure.figsize"] = (5 * columns, 5 * row_nums)
    plt.imshow(c4)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.savefig(os.path.join(saved_loc, "cam_gb.jpg"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='resnet50')
    parser.add_argument("--img-dir", type=str, default='images')
    parser.add_argument("--img-name", type=str, default='both.png')
    parser.add_argument("--finalconv-name", type=str, default='layer1')
    parser.add_argument("--active-layer", type=int, default=254)
    parser.add_argument("--row-nums", type=int, default=20)
    parser.add_argument("--saved-loc", type=str, default='./layer1_active_254')

    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=30)

    parser.add_argument("--target-list", nargs="+", default=[i for i in range(900)])

    args = parser.parse_args()

    save_adv_cam(args)
