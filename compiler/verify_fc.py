import numpy as np

# Load inputs
x = np.load("x_fc.npy")         # Shape: (36,)
W = np.load("W_fc.npy")         # Shape: (10, 36)
b = np.load("b_fc.npy")         # Shape: (10,)
sim = np.load("../simulator/sim_output.npy")  # Shape: (10,)

# Compute NumPy reference
y = np.dot(W, x) + b
y_relu = np.maximum(0, y)

# Round if needed (for int simulation)
y_relu = y_relu.astype(int)

# Show both results
print("🔬 NumPy Reference Output:")
print(y_relu)

print("\n🧠 Simulator Output:")
print(sim)

# Compare
if np.array_equal(y_relu, sim):
    print("\n✅ Match! FC layer simulation is correct.")
else:
    print("\n❌ Mismatch! Check your instruction sequence or memory inputs.")
    print("Difference:", y_relu - sim)

