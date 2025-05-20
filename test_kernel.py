import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(torch.cuda.get_device_name(0))
    x = torch.ones(10, device=device)
    y = torch.ones(10, device=device)
    print("Soma:", x + y)
else:
    print("GPU não disponível.")
