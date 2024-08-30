import torch.utils.data as data
import torch
import h5py
import os
from PIL import Image
import numpy as np

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        dataset = h5py.File(file_path)


        self.image1 = dataset.get("image1")

        self.image2= dataset.get("image2")
        self.GT = dataset.get("hr")




    #####必要函数
    def __getitem__(self, index):
        return torch.from_numpy(self.image1[index, :, :, :]).float(), \
               torch.from_numpy(self.image2[index, :, :, :]).float(), \
               torch.from_numpy(self.GT[index, :, :, :]).float()


                   #####必要函数
    def __len__(self):
        return self.image1.shape[0]
    
    
class DatasetFromHdf5_real(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5_real, self).__init__()
        dataset = h5py.File(file_path)


        self.GT = dataset.get("hr")


    #####必要函数
    def __getitem__(self, index):
        return torch.from_numpy(self.GT[index, :, :, :]).float()


                   #####必要函数
    def __len__(self):
        return self.GT.shape[0]
    
class DatasetFromFolder_REAL3(data.Dataset):
    def __init__(self, folder_path1, folder_path2, folder_path3):
        super(DatasetFromFolder, self).__init__()
        self.image_paths1 = [os.path.join(folder_path1, file) for file in os.listdir(folder_path1) if file.endswith('.tif')]
        self.image_paths2 = [os.path.join(folder_path2, file) for file in os.listdir(folder_path2) if file.endswith('.tif')]
        self.GTs = [os.path.join(folder_path3, file) for file in os.listdir(folder_path3) if file.endswith('.tif')]

    def __len__(self):
        return len(self.image_paths1)
    
    def __getitem__(self, index):
        image_path1 = self.image_paths1[index]
        image_path2 = self.image_paths2[index]
        GT = self.GTs[index]
        image_name1 = os.path.basename(image_path1)
        image_name2 = os.path.basename(image_path2)
        GT_name = os.path.basename(GT)
        image1 = Image.open(image_path1)
        image2 = Image.open(image_path2)
        gt = Image.open(GT)
        image1 = np.array(image1)
        image2 = np.array(image2)
        gt = np.array(gt)

    # 如果需要进行预处理，可以在这里添加代码

        return image1, image_name1, image2, image_name2, gt, GT_name

class DatasetFromHdf5_real(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5_real, self).__init__()
        dataset = h5py.File(file_path)


        self.GT = dataset.get("hr")


    #####必要函数
    def __getitem__(self, index):
        return torch.from_numpy(self.GT[index, :, :, :]).float()


                   #####必要函数
    def __len__(self):
        return self.GT.shape[0]
    
    
class DatasetFromFolder(data.Dataset):
    def __init__(self, folder_path1, folder_path2):
        super(DatasetFromFolder, self).__init__()
        self.image_paths1 = [os.path.join(folder_path1, file) for file in os.listdir(folder_path1) if file.endswith('.tif')]
        self.image_paths2 = [os.path.join(folder_path2, file) for file in os.listdir(folder_path2) if file.endswith('.tif')]


    def __len__(self):
        return len(self.image_paths1)
    
    def __getitem__(self, index):
        image_path1 = self.image_paths1[index]
        image_path2 = self.image_paths2[index]

        image_name1 = os.path.basename(image_path1)
        image_name2 = os.path.basename(image_path2)

        image1 = Image.open(image_path1)
        image2 = Image.open(image_path2)
   
        image1 = np.array(image1)
        image2 = np.array(image2)
 

    # 如果需要进行预处理，可以在这里添加代码

        return image1, image_name1, image2, image_name2
