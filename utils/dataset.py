import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random

class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('image', 'label').split('.')[0]+"_mask.png"
        # 读取训练图片和标签图片
        
        image = cv2.imread(image_path, 0)
        label = cv2.imread(label_path, 0)
      
        # 将数据转为单通道的图片
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)



# import torch.utils.data as data
# import PIL.Image as Image
# from sklearn.model_selection import train_test_split
# import os
# import random
# import numpy as np
# from skimage.io import imread
# import cv2
# from glob import glob
# import imageio
# import torch 
# from torchvision.transforms import transforms



# class esophagusDataset(data.Dataset):
#     def __init__(self, state, transform=None, target_transform=None):
#         self.state = state
#         self.train_root = "/data/wangkun/data_sta_all/train_data"
#         self.val_root = "/data/wangkun/data_sta_all/test_data"
#         self.test_root = self.val_root
#         self.pics,self.masks = self.getDataPath()
#         self.transform = transform
#         self.target_transform = target_transform

#     def getDataPath(self):
#         assert self.state =='train' or self.state == 'val' or self.state == 'test'
#         if self.state == 'train':
#             root = self.train_root
#         if self.state == 'val':
#             root = self.val_root
#         if self.state == 'test':
#             root = self.test_root
#         pics = []
#         masks = []
#         n = len(os.listdir(root)) // 2  # 因为数据集中一套训练数据包含有训练图和mask图，所以要除2
#         for i in range(n):
#             img = os.path.join(root, "%05d.png" % i)  # liver is %03d
#             mask = os.path.join(root, "%05d_mask.png" % i)
#             pics.append(img)
#             masks.append(mask)
#             #imgs.append((img, mask))
#         return pics,masks

#     def __getitem__(self, index):
#         #x_path, y_path = self.imgs[index]
#         x_path = self.pics[index]
#         y_path = self.masks[index]
#         # origin_x = Image.open(x_path)
#         # origin_y = Image.open(y_path)
#         origin_x = cv2.imread(x_path)
#         origin_y = cv2.imread(y_path,cv2.COLOR_BGR2GRAY)
#         if self.transform is not None:
#             img_x = self.transform(origin_x)
#         if self.target_transform is not None:
#             img_y = self.target_transform(origin_y)
#         return img_x, img_y,x_path,y_path

#     def __len__(self):
#         return len(self.pics)




if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    # x_transforms = transforms.Compose([
    #     transforms.ToTensor(),  # -> [0,1]
    #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
    # ])

    # # mask只需要转换为tensor
    # y_transforms = transforms.ToTensor()

    # train_dataset = esophagusDataset( r'train', transform=x_transforms, target_transform=y_transforms)
    # print("数据个数：", len(train_dataset))
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    #                                             batch_size=4, 
    #                                             shuffle=True)













if __name__ == "__main__":
    isbi_dataset = ISBI_Loader("/data/wangkun/dataset_96/train_96")
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=4, 
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)