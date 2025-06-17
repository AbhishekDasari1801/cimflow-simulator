import numpy as np

N = 5  # Number of layers
input_dim = 64
output_dims = [128, 64, 64, 32, 16]  # Output size for each layer

# Save input
x = np.random.randint(0, 5, (input_dim,))
np.save("input_x.npy", x)

for i in range(N):
    W = np.random.randint(0, 5, (output_dims[i], input_dim))
    b = np.array([np.random.randint(0, 10)])
    np.save(f"W{i+1}.npy", W)
    np.save(f"b{i+1}.npy", b)
    input_dim = output_dims[i]  # next input size = current output

