import numpy as np
import json
import jsonlines
import os
from os import listdir
from os.path import isfile, join
import torch
from torch import optim
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import cv2
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from torchvision import transforms
import seaborn as sns

# THINGS YOU MIGHT WANT TO CHANGE
device = torch.device('cuda:1')
BATCH_SIZE = 4
LEARNING_RATE = 0.001 #3e-4
folder_path = "/srv/data/lt2326-h25/a1/images"

# desired length of image and given length
given_length = 2048
length_img = 512
devider = given_length/length_img


print("\n--preparing data--")
# list of all avalaible images
avlbl_images = [im for im in listdir(folder_path) if isfile(join(folder_path, im))]

#I'm creating dictionary of dictionaries where key is image ID, and the value is dictionary. The dictionary is build of keys:ordinal number for each character in the image and values: bbox cordinates
#{imageID: {nr_of_annot : [bbox_coordinates]}
#{'0000172.jpg': {0: [140, 897, 22, 38],

with jsonlines.open("train.jsonl") as reader:
    #I'm getting info from this file
    train_data = reader
    cordinates_dict = {}
    for i, line in enumerate(train_data):
        if i < 10000:
            for desired_file in avlbl_images:
                #checking if file from big data set is in our images folder
                if line["file_name"] == desired_file:
                    #here I'm implementing number for each character in the image
                    #sub dict that will be value for image ID
                    all_char_in_ann_lsols = []
                    #iterating through annotations
                    for ann in (line["annotations"]):
                        #iterating through characters
                        for char in ann:
                            if char["is_chinese"] == True:
                                adjusted_bbox_intgrs = []
                                #changing cordinates from float to integers
                                for float_number in char["adjusted_bbox"]:
                                    adjusted_bbox_intgrs.append(int(float_number))
                                all_char_in_ann_lsols.append(adjusted_bbox_intgrs)
                    
                        cordinates_dict[desired_file] = all_char_in_ann_lsols
# number of images both in json data and in our folder is 845
# print(len(cordinates_dict))

#Creating tuple for each image: (original image , 01 encoded characters) and keeping in in the dictonary
# {file_name:(input, output)}

print("\n--creating golden labels--")

source_output_dict = {}
for nr, file_name in enumerate(cordinates_dict):
    if nr<1000:
        #Here I'm preapering input of an object - tensor of an img: torch.Size([3, 2048, 2048])

        
        img_cv2array = cv2.imread("images/" + file_name)
        img_cv2array = cv2.resize(img_cv2array, (length_img, length_img), interpolation=cv2.INTER_AREA)
        img_pytorch_tensor = torch.from_numpy(img_cv2array)
        img_pytorch_tensor_prmtd = img_pytorch_tensor.permute(2, 0, 1)

        #Here I'm preapering binary matrix where 0-represents a pixel which is not a chineese character and 1- is in a bbx of chineese character  
        pxls_lst = np.zeros((length_img, length_img), dtype=np.uint8)

        scale = length_img / 2048
        pxls_lst = np.zeros((length_img, length_img), dtype=np.uint8)

        
        for x_min, y_min, w, h in cordinates_dict[file_name]:
            x1 = int(x_min * scale)
            y1 = int(y_min * scale)
            x2 = int((x_min + w) * scale)
            y2 = int((y_min + h) * scale)
            pxls_lst[y1:y2, x1:x2] = 1

                       
        output_tnsr = torch.tensor(pxls_lst, dtype=torch.float32)
        output_tnsr = output_tnsr.unsqueeze(0)
        source_output_dict[file_name] = (img_pytorch_tensor_prmtd, output_tnsr)

def split_dict(big_dict, train_ratio=0.7, test_ratio=0.15, dev_ratio=0.15):
    
    random.seed(42)
    keys_list = list(big_dict.keys())
    random.shuffle(keys_list)

    total_len = len(keys_list)
    intg_train = int(total_len*train_ratio)
    intg_dev = int(total_len*dev_ratio)

    # test = rest
    
    train_keys = keys_list[:intg_train]
    dev_keys = keys_list[intg_train:(intg_dev+intg_train)]
    test_keys = keys_list[(intg_train+intg_dev):]

    
    train_dict = {x: big_dict[x] for x in train_keys}
    dev_dict = {x: big_dict[x] for x in dev_keys}
    test_dict = {x: big_dict[x] for x in test_keys}

    # return train_dict
    return train_dict, dev_dict, test_dict

train_dict, dev_dict, test_dict = split_dict(source_output_dict)

print("\n--creating datasets for dataloader--")

class chin_char_ds(Dataset):
    def __init__(self, main_dict):
        self.main_dict = list(main_dict.values())
    def __len__(self):
        return len(self.main_dict)
    def __getitem__(self, idx):
        img, mask = self.main_dict[idx][0], self.main_dict[idx][1]
    
        # Convert to float and normalize to [0,1]
        img = img.float() / 255.0
        mask = mask.float()
    
        return img, mask

train_ds  = chin_char_ds(train_dict)
dev_ds  = chin_char_ds(dev_dict)
test_ds  = chin_char_ds(test_dict)

# print(train_ds.__len__())
# print(train_ds.__getitem__(0))

train_dataloader = DataLoader(dataset=train_ds,
                              # pin_memory=False,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

dev_dataloader = DataLoader(dataset=dev_ds,
                            # pin_memory=False,
                            batch_size=BATCH_SIZE,
                            shuffle=True)

test_dataloader = DataLoader(dataset=test_ds,
                            # pin_memory=False,
                            shuffle=True)

class My_Model(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.down1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(2))
        
        self.down2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(2))

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv_up1 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1),
                                      nn.ReLU(inplace=True))

        self.up2 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.conv_up2 = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1),
                                      nn.ReLU(inplace=True))

        self.output = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.up1(x2)
        x4 = self.conv_up1(x3)
        x5 = self.up2(x4)
        out = self.output(x5)
        return out

my_model = My_Model().to(device)
optimizer_mm = optim.AdamW(my_model.parameters(), lr=LEARNING_RATE)
criterion_mm = nn.BCEWithLogitsLoss()

print("\n--training first model: My Model--\n")


EPOCHSmm = 3
train_losses_mm = []
val_losses_mm = []


for epoch in tqdm(range(EPOCHSmm)):

    my_model.train()
    train_running_loss = 0
    train_running_dc = 0

    for idx, img_mask in enumerate(tqdm(train_dataloader, position=0, leave=True)):
        img = img_mask[0].float().to(device)
        mask = (img_mask[1]>0).float().to(device)

        y_pred = my_model(img)
        optimizer_mm.zero_grad()

        loss = criterion_mm(y_pred, mask)

        train_running_loss += loss.item()

        loss.backward()
        optimizer_mm.step()

    train_loss = train_running_loss / (idx + 1)

    train_losses_mm.append(train_loss)

    my_model.eval()
    val_running_loss = 0
    val_running_dc = 0
    
    with torch.no_grad():
        for idx, img_mask in enumerate(tqdm(dev_dataloader, position=0, leave=True)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_pred = my_model(img)
            loss = criterion_mm(y_pred, mask)
            
            val_running_loss += loss.item()

        val_loss = val_running_loss / (idx + 1)

    val_losses_mm.append(val_loss)

    print("-" * 30)
    print(f"Training Loss EPOCH {epoch + 1}: {train_loss:.4f}")
    print("\n")
    print(f"Validation Loss EPOCH {epoch + 1}: {val_loss:.4f}")
    print("-" * 30)

trained_model_mm = My_Model().to(device)

test_running_loss_mm = 0

with torch.no_grad():
    for idx, img_mask in enumerate(tqdm(test_dataloader, position=0, leave=True)):
        img = img_mask[0].float().to(device)
        mask = img_mask[1].float().to(device)

        y_pred = trained_model_mm(img)
        loss = criterion_mm(y_pred, mask)

        test_running_loss_mm += loss.item()
        
trained_model_mm.eval()
golden_label_lst, prediction_lst = [], []

with torch.no_grad():
    for img, mask in test_ds:
        img = img.to(device)
        pred = trained_model_mm(img.unsqueeze(0)).cpu() 
        golden_label_lst.append(mask)
        prediction_lst.append(pred)
        
pred_labels = [ (pred > 0.5).long().squeeze().numpy() for pred in prediction_lst ]
golden_flat = np.concatenate([g.numpy().flatten() for g in golden_label_lst])
pred_flat   = np.concatenate([p.flatten() for p in pred_labels])
print("\nF1 score for my model")
print(f1_score(golden_flat, pred_flat, average="weighted"))

print("\ntraining second model: à la U-net")

#a la U-net model
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                            nn.ReLU(inplace=True)
                            )
    def forward(self, x):
        return self.conv_op(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)
        return down, p

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)

class UNet(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)

        self.bottle_neck = DoubleConv(256, 512)

        self.up_convolution_1 = UpSample(512, 256)
        self.up_convolution_2 = UpSample(256, 128)
        self.up_convolution_3 = UpSample(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)

        b = self.bottle_neck(p3)

        up_1 = self.up_convolution_1(b, down_3)
        up_2 = self.up_convolution_2(up_1, down_2)
        up_3 = self.up_convolution_3(up_2, down_1)

        out = self.out(up_3)
        return out

model = UNet(in_channels=3, num_classes=1).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

def dice_coefficient(prediction, target, epsilon=1e-07):
    prediction_copy = prediction.clone()

    prediction_copy[prediction_copy >= 0.5] = 1
    prediction_copy[prediction_copy < 0.5 ] = 0
    # prediction_copy[prediction_copy < 0] = 0
    # prediction_copy[prediction_copy > 0] = 1

    
    intersection = abs(torch.sum(prediction_copy * target))
    union = abs(torch.sum(prediction_copy) + torch.sum(target))
    dice = (2. * intersection + epsilon) / (union + epsilon)
    
    return dice



EPOCHS = 3

train_losses = []
train_dcs = []
val_losses = []
val_dcs = []


for epoch in tqdm(range(EPOCHS)):
    
    model.train()
    train_running_loss = 0
    train_running_dc = 0

    for idx, img_mask in enumerate(tqdm(train_dataloader, position=0, leave=True)):
        img = img_mask[0].float().to(device)
        # mask = img_mask[1].float().to(device)
        mask = (img_mask[1]>0).float().to(device)


        y_pred = model(img)
        optimizer.zero_grad()

        dc = dice_coefficient(y_pred, mask)
        loss = criterion(y_pred, mask)

        train_running_loss += loss.item()
        train_running_dc += dc.item()

        loss.backward()
        optimizer.step()

    train_loss = train_running_loss / (idx + 1)
    train_dc = train_running_dc / (idx + 1)

    train_losses.append(train_loss)
    train_dcs.append(train_dc)

    model.eval()
    val_running_loss = 0
    val_running_dc = 0
    
    with torch.no_grad():
        for idx, img_mask in enumerate(tqdm(dev_dataloader, position=0, leave=True)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_pred = model(img)
            loss = criterion(y_pred, mask)
            dc = dice_coefficient(y_pred, mask)
            
            val_running_loss += loss.item()
            val_running_dc += dc.item()

        val_loss = val_running_loss / (idx + 1)
        val_dc = val_running_dc / (idx + 1)

    val_losses.append(val_loss)
    val_dcs.append(val_dc)


    print("-" * 30)
    print(f"Training Loss EPOCH {epoch + 1}: {train_loss:.4f}")
    print(f"Training DICE EPOCH {epoch + 1}: {train_dc:.4f}")
    print("\n")
    print(f"Validation Loss EPOCH {epoch + 1}: {val_loss:.4f}")
    print(f"Validation DICE EPOCH {epoch + 1}: {val_dc:.4f}")
    print("-" * 30)

trained_model = UNet(in_channels=3, num_classes=1).to(device)

test_running_loss = 0
test_running_dc = 0

with torch.no_grad():
    for idx, img_mask in enumerate(tqdm(test_dataloader, position=0, leave=True)):
        img = img_mask[0].float().to(device)
        mask = img_mask[1].float().to(device)

        y_pred = trained_model(img)
        loss = criterion(y_pred, mask)
        dc = dice_coefficient(y_pred, mask)

        test_running_loss += loss.item()
        test_running_dc += dc.item()

    test_loss = test_running_loss / (idx + 1)
    test_dc = test_running_dc / (idx + 1)

print("dice test accuracy")
print(test_dc)
