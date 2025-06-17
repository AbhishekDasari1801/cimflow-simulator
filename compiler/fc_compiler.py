import numpy as np

# Load output from CNN
x = np.load("cnn_output.npy").flatten()  # shape: (36,)

# Define FC weights and bias (10 output classes)
W = np.random.randint(0, 5, (10, 36))
b = np.random.randint(0, 5, (10,))

np.save("W_fc.npy", W)
np.save("b_fc.npy", b)
np.save("x_fc.npy", x)

isa = []
# Assume x is loaded to memory starting at addr 500
# W and b are in memory at 600 and 900 respectively

# Load each weight row and compute dot(W[i], x)
for i in range(10):
    isa.append(f"# FC neuron {i}")
    isa.append("CLEAR 3")  # R3 = 0 accumulator
    for j in range(36):
        isa.append(f"LOAD 1 {500 + j}")      # R1 = x[j]
        isa.append(f"LOAD 2 {600 + i*36 + j}")  # R2 = W[i][j]
        isa.append("MUL 4 1 2")              # R4 = R1 * R2
        isa.append("ADD 3 3 4")              # R3 += R4

    # Add bias and apply ReLU
    isa.append(f"LOAD 1 {900 + i}")
    isa.append("ADD 4 3 1")
    isa.append("RELU 5 4")
    isa.append(f"STORE 5 {1000 + i * 10}")

with open("program_fc.isa", "w") as f:
    f.write("\n".join(isa))

print("✅ FC ISA generation complete.")

