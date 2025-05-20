import torch
print(torch.version.hip)  # Verifica se está com HIP/ROCm
print(torch.cuda.is_available())  # Verifica se há GPU visível
print(torch.cuda.get_device_name(0))  # Nome da GPU detectada
