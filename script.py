import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Tamanho dos tensores
size = (8192, 8192)

# Número de iterações
iterations = 100

# Timer
start = time.time()

# Loop de stress
for i in range(iterations):
    a = torch.rand(size, device=device)
    b = torch.rand(size, device=device)
    c = torch.matmul(a, b)  # operação pesada
    c = c.relu()            # mais uma operação
    del a, b, c             # libera memória a cada ciclo
    torch.cuda.empty_cache()

    if (i + 1) % 10 == 0:
        print(f"Iteração {i + 1}/{iterations} concluída")

end = time.time()
print(f"Tempo total: {end - start:.2f} segundos")
