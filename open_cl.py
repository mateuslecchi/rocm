import pyopencl as cl
import numpy as np
import time

# Cria contexto e fila de comandos (assume primeira GPU AMD disponível)
platforms = cl.get_platforms()
gpu_devices = platforms[0].get_devices(device_type=cl.device_type.GPU)
ctx = cl.Context(devices=gpu_devices)
queue = cl.CommandQueue(ctx)

print(f"Dispositivo em uso: {gpu_devices[0].name}")

# Kernel OpenCL para multiplicação de vetores (simples, mas intenso em volume)
kernel_code = """
__kernel void vec_add(__global float* a, __global float* b, __global float* c) {
    int i = get_global_id(0);
    c[i] = a[i] * b[i];
}
"""

# Compila o kernel
program = cl.Program(ctx, kernel_code).build()

# Tamanho do vetor
size = 100_000_000  # 100 milhões de elementos (≈ 400 MB por vetor)
iterations = 50     # Número de repetições

print(f"Iniciando stress test com {iterations} iterações e vetor de {size:,} floats (~1.2 GB por iteração)\n")

# Aloca buffers
a_np = np.random.rand(size).astype(np.float32)
b_np = np.random.rand(size).astype(np.float32)
c_np = np.empty_like(a_np)

a_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a_np)
b_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b_np)
c_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, c_np.nbytes)

start_time = time.time()

for i in range(iterations):
    program.vec_add(queue, (size,), None, a_buf, b_buf, c_buf)
    cl.enqueue_copy(queue, c_np, c_buf)
    queue.finish()

    if (i + 1) % 10 == 0:
        print(f"Iteração {i + 1}/{iterations} completa")

end_time = time.time()

print(f"\nTeste de stress finalizado em {end_time - start_time:.2f} segundos.")
