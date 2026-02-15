#%%
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
#%%
class MyDataset(Dataset): # 继承 Dataset 基类，重写 __init__, __getitem__, __len__ 三个方法
    def __init__(self, root_dir, img_dir, label_dir): 
        self.root_dir = root_dir # 根目录
        self.img_dir = img_dir # 图片目录
        self.label_dir = label_dir # 标签目录
        self.img_path = os.listdir(os.path.join(self.root_dir, self.img_dir)) # 获取图片列表

    def __getitem__(self, idx):
        img_name = self.img_path[idx] # 根据索引获取图片文件名
        img_item_path = os.path.join(self.root_dir, self.img_dir, img_name)
        img = Image.open(img_item_path)

        label_name = img_name.split('.')[0] + '.txt' # 根据图片文件名生成对应的标签文件名
        label_item_path = os.path.join(self.root_dir, self.label_dir, label_name) # 标签文件路径
        
        with open(label_item_path, 'r') as f:
            label = f.read()

        return img, label

    def __len__(self):
        return len(self.img_path)
# %%
root_dir = 'dataset/hymenoptera_data/train'
ants_dataset = MyDataset(root_dir, 'ants_image', 'ants_label')
bees_dataset = MyDataset(root_dir, 'bees_image', 'bees_label')
# %%
train_dataset = ants_dataset + bees_dataset
# %%
