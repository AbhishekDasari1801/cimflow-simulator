import numpy as np

# Load data
input_image = np.load("input_image.npy")
kernel = np.load("kernel.npy")
bias = int(np.load("bias.npy")[0])
sim_output = np.load("../simulator/cnn_output.npy")

# Reference output using NumPy
output_size = 6
output_ref = np.zeros((output_size, output_size), dtype=int)

for i in range(output_size):
    for j in range(output_size):
        patch = input_image[i:i+3, j:j+3]
        val = np.sum(patch * kernel) + bias
        output_ref[i, j] = max(0, val)

# Print both outputs
print("\n🔎 NumPy Reference Output:")
print(output_ref)

print("\n🧠 Simulator Output:")
print(sim_output)

# Compare
if np.array_equal(output_ref, sim_output):
    print("\n✅ Match! Simulator is functionally correct.")
else:
    print("\n❌ Mismatch detected! Debug patch-wise.")
    diff = output_ref - sim_output
    print("Difference:\n", diff)

