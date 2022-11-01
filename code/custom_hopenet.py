import sys, os, argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
import glob
import pickle
import os

import datasets, hopenet, utils
from mtcnn import MTCNN
import cv2


def head_pose(input):
    snapshot_path='dhp/hopenet_robust_alpha1.pkl'
    cudnn.enabled = True
    gpu = 0
    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)

    transformations = transforms.Compose([transforms.Resize(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    model.cuda(gpu)

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    yaw_error = .0
    pitch_error = .0
    roll_error = .0

    l1loss = torch.nn.L1Loss(size_average=False)
    image_pose = {}
    image_np_arr = []
    index = 0
    img_list = input
    for img_np in img_list:
        img = Image.fromarray(img_np)
        img = transformations(img)
        img=img.unsqueeze(0)
        
        images = Variable(img).cuda(gpu)
    
        yaw, pitch, roll = model(images)
        # Binned predictions
        _, yaw_bpred = torch.max(yaw.data, 1)
        _, pitch_bpred = torch.max(pitch.data, 1)
        _, roll_bpred = torch.max(roll.data, 1)

        # Continuous predictions
        yaw_predicted = utils.softmax_temperature(yaw.data, 1)
        pitch_predicted = utils.softmax_temperature(pitch.data, 1)
        roll_predicted = utils.softmax_temperature(roll.data, 1)

        yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 99
        roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 99

    
        pitch = pitch_predicted[0]
        yaw = -yaw_predicted[0] 
        roll = roll_predicted[0] 
        #save to dict
        image_pose[index] = {'pitch': pitch, 'yaw': yaw, 'roll':roll}
        image_np_arr[index] = img_np
        index +=1
    return image_pose, image_np_arr

def find_similar(image_coords_, avatar_coords_):

    dissimilarity = {}
    for i in image_coords_:
        for a in avatar_coords_:
            x = image_coords_[i].get('pitch')
            y = image_coords_[i].get('yaw')
            z = image_coords_[i].get('roll')
            x_fake = avatar_coords_[a].get('pitch')
            y_fake = avatar_coords_[a].get('yaw')
            z_fake = avatar_coords_[a].get('roll')
            
            real = np.array([x,y,z])
            fake = np.array([x_fake, y_fake, z_fake])
            dissimilarity[a] = np.linalg.norm((real-fake)*2)
            
        key_min = min(dissimilarity.keys(), key=(lambda k: dissimilarity[k]))
        #return index of images with max similarity
        original_image = i
        avatar_image = key_min
    return original_image, avatar_image

def match_head_pose(image_np, avatar_np):
    image_coords, image_np_arr = head_pose(image_np)
    avatar_coords, avatar_np_arr = head_pose(avatar_np)

    matching_original_image, matching_avatar_image = find_similar(image_coords, avatar_coords)

    #get np_array of the matching images
    original_image = image_np_arr[matching_original_image]
    avatar_image = avatar_np_arr[matching_avatar_image]
    
    return original_image, avatar_image
