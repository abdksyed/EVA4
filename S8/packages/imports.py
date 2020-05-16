import torch
import matplotlib.pyplot as plt
import data
import model
import summary
import test
import train
import utilis
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For Reproducibility
torch.manual_seed(1)

if cuda:
    torch.cuda.manual_seed(1)
print(torch.cuda.get_device_name(0))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
