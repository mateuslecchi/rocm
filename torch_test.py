import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.hip)
print(torch.cuda.get_device_name(0))
