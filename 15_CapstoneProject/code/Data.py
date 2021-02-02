from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import time


class DepMaskDataset(Dataset):
    def __init__(self, file_list):

        self.to_tensor = transforms.ToTensor()
        self.fg_norm = transforms.Normalize(mean=[0.56414794921875, 0.5432481169700623, 0.5214933753013611],
                                            std=[0.208669513463974, 0.213966503739357, 0.2230370044708252])
        self.bg_norm = transforms.Normalize(mean=[0.5810659527778625, 0.5633341670036316, 0.5403051972389221],
                                            std=[0.18558748066425323, 0.19134603440761566, 0.20390349626541138])
        self.mask_norm = transforms.Normalize(mean=0.09639497846364975,
                                              std=0.2823259234428406)
        self.depth_norm = transforms.Normalize(mean=0.40525129437446594,
                                               std=0.24363896250724792)
        self.file_name_list = file_list
        self.fgbg = 'Data/Dataset/fg_bg_/fg_bg/'
        self.bg = 'Data/Dataset/bg_/bg_192/'
        self.mask = 'Data/Dataset/mask_/mask/'
        self.depth = 'Data/Dataset/depth_/depth/'
        self.len = len(self.file_name_list)

    def __getitem__(self, index):
        # stuff

        fg = Image.open(self.fgbg+self.file_name_list[index]).convert('RGB')
        bg = Image.open(
            self.bg+self.file_name_list[index][0:8]+'.jpg').convert('RGB')
        mask = Image.open(self.mask+self.file_name_list[index]).convert('L')
        mask_arr = np.array(mask)
        mask_arr[mask_arr >= 150] = 255
        mask_arr[mask_arr < 150] = 0
        mask = Image.fromarray(mask_arr).convert('L')
        depth = Image.open(self.depth+self.file_name_list[index]).convert('L')

        fg = self.to_tensor(fg)
        bg = self.to_tensor(bg)
        mask = self.to_tensor(mask)
        depth = self.to_tensor(depth)

        fg = self.fg_norm(fg)
        bg = self.bg_norm(bg)
        mask = self.mask_norm(mask)
        depth = self.depth_norm(depth)

        return {'fgbg': fg, 'bg': bg, 'mask': mask, 'depth': depth}

    def __len__(self):
        return self.len  # of how many data(images) you have
