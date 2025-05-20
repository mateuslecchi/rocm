import torch

# Check available devices
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))

# Use "cuda" instead of "hip:0" for AMD GPUs with ROCm
# ROCm is recognized as "cuda" in PyTorch's API
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Create tensor
size = (1000, 1000)
try:
    a = torch.rand(size, device=device)
    print("Tensor created successfully:", a.shape)
except Exception as e:
    print(f"Error with cuda device, falling back to CPU: {e}")
    # Fallback to CPU if GPU fails
    device = torch.device("cpu")
    print("Using fallback device:", device)
    a = torch.rand(size, device=device)
    print("Tensor created successfully on CPU:", a.shape)