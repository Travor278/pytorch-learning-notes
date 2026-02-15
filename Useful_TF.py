# 关注输入输出的维度，数据类型，数值范围
# 多看官方文档，理解每个函数的功能和参数
# 关注方法需要什么参数 
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open(r"dataset\hymenoptera_data\train\ants_image\0013035.jpg")
print(img)

# Totensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([6, 3, 2], [9, 3, 5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm, 2)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
# img PIL -> resize -> img_resize PIL -> totensor -> img_resize tensor
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize) 
writer.add_image("Resize", img_resize, 0)
print(img_resize)

# Compose - resize - 2
trans_resize_2 = transforms.Resize(512)
# PIL -> resize -> PIL -> totensor -> tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)

# RandomCrop
trans_random = transforms.RandomCrop((50,100))
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_random = trans_compose_2(img)
    writer.add_image("RandomCrop", img_random, i)

writer.close()