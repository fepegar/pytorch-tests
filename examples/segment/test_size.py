import torch
from unet_github import UNet3D
from pytorch_modelsize import SizeEstimator

model = UNet3D(
    in_channels=1, n_classes=2,
)
i = torch.rand(1, 1, 97, 115, 97)  # 2 mm

with torch.no_grad():
    se = SizeEstimator(model, input_size=i.shape)
    print(se.estimate_size())
    print(se.param_bits) # bits taken up by parameters
    print(se.forward_backward_bits) # bits stored for forward and backward
    print(se.input_bits) # bits for input
