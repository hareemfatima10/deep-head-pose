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
from torchvision.models.vision_transformer import ImageClassification
from mtcnn import MTCNN
import cv2


def head_pose(input):
    snapshot_path='/content/hopenet_robust_alpha1.pkl'
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
    image_pose = []
    image_np_arr = []
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
        image_pose.append((pitch,yaw,roll))
        #image_pose[index] = {'pitch': pitch, 'yaw': yaw, 'roll':roll}
        image_np_arr.append(img_np)
    return image_pose, image_np_arr

def find_similar(image_coords_, avatar_coords_):
    dissimilarity = {}
    similar_pose =[]
    for i in image_coords_:
        for a in avatar_coords_:
            x = i[0]
            y = i[1]
            z = i[2]
            x_fake = a[0]
            y_fake = a[1]
            z_fake = a[2]
            
            real = np.array([x,y,z])
            fake = np.array([x_fake, y_fake, z_fake])
            dissimilarity[a] = np.linalg.norm((real-fake)*2)
            
        key_min = min(dissimilarity.keys(), key=(lambda k: dissimilarity[k]))
        #return index of images with max similarity
        similar_pose.append((image_coords_.index(i),avatar_coords_.index(key_min)))
    return similar_pose

def match_head_pose(image_np, avatar_np):
    images = []
    image_coords, image_np_arr = head_pose(image_np)
    avatar_coords, avatar_np_arr = head_pose(avatar_np)
    similar_pose = find_similar(image_coords, avatar_coords)
    for i in similar_pose:
      images.append((image_np_arr[i[0]],avatar_np_arr[i[1]]))
    
    return images
