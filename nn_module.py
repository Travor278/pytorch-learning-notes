import torch
import torch.nn as nn

class Moli(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output
    
moli = Moli()
x = torch.tensor(1.0)
output = moli(x)
print(output)