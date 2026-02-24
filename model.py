from torch import nn
import torch

# 搭建神经网络
class Moli(nn.Module):
    def __init__(self):
        super(Moli, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),  # [b, 3, 32, 32] -> [b, 32, 32, 32]
            nn.MaxPool2d(2),            # [b, 32, 32, 32] -> [b, 32, 16, 16]
            nn.Conv2d(32, 32, 5, 1, 2), # [b, 32, 16, 16] -> [b, 32, 16, 16]
            nn.MaxPool2d(2),            # [b, 32, 16, 16] -> [b, 32, 8, 8]
            nn.Conv2d(32, 64, 5, 1, 2), # [b, 32, 8, 8] -> [b, 64, 8, 8]
            nn.MaxPool2d(2),            # [b, 64, 8, 8] -> [b, 64, 4, 4]
            nn.Flatten(),               # [b, 64, 4, 4] -> [b, 1024]
            nn.Linear(1024, 64),        # [b, 1024] -> [b, 64]
            nn.Linear(64, 10)           # [b, 64] -> [b, 10]
        )

    def forward(self, x):
        return self.model(x)
    
if __name__ == "__main__":
    moli = Moli()
    input = torch.ones((64, 3, 32, 32))
    output = moli(input)
    print(output.shape)