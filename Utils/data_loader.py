
import numpy as np
import pandas as pd
import torch
import cv2


class ImageDataLoader(torch.utils.data.Dataset):
    def __init__(self,csv,train=True):
        self.csv=pd.read_csv(csv)
        self.img_size=28
        # print(self.csv['image_names'][:5])
        self.train=train
        text="pixel"
        self.images=torch.zeros((self.csv.shape[0],1))
        for i in range(1,785):
            temp_text=text+str(i)
            temp=self.csv[temp_text]
            temp=torch.FloatTensor(temp).unsqueeze(1)
            self.images=torch.cat((self.images,temp),1)
        self.labels=self.csv['label']
        self.images=self.images[:,1:]
        self.images=self.images.view(-1,28,28)
        
    def __getitem__(self,index):
        img=self.images[index]
        img=img.numpy()
        img=cv2.resize(img,(self.img_size,self.img_size))
        tensor_image=torch.FloatTensor(img)
        tensor_image=tensor_image.unsqueeze(0)
        tensor_image/=255.
        if self.train:
            return tensor_image,self.labels[index]
        else:
            return tensor_image
    def __len__(self):
        return self.images.shape[0]