import numpy as np

W = np.load("W.npy")
x = np.load("x.npy")
b = np.load("b.npy")

# Simulate tile-wise dot + accumulation (matching simulator logic)
tile_size = 256
y_partial_sum = 0
for i in range(0, W.shape[0], tile_size):
    W_tile = W[i:i+tile_size, :]
    y_partial_sum += np.sum(W_tile @ x)  # Accumulate scalar sum across all tiles

# Bias and RELU
y_final = y_partial_sum + b[0]
y_relu = max(0, y_final)

# Load simulator output
sim_output = np.load("../simulator/sim_output.npy")[0]

print(f"\n✅ Simulator: {sim_output}")
print(f"✅ NumPy Ref : {y_relu}")

if sim_output == y_relu:
    print("✅ Match! Simulation is functionally correct.")
else:
    print("❌ Mismatch! Something to debug.")

