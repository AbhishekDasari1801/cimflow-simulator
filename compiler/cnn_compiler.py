import numpy as np

# Config
input_size = 8
kernel_size = 3
stride = 1
padding = 0

output_size = (input_size - kernel_size + 2 * padding) // stride + 1  # 6x6

# Create dummy input image and weight
input_image = np.random.randint(0, 5, (input_size, input_size))
kernel = np.random.randint(0, 5, (kernel_size, kernel_size))
bias = np.random.randint(0, 5)

# Save input, kernel, and bias for simulator
np.save("input_image.npy", input_image)
np.save("kernel.npy", kernel)
np.save("bias.npy", np.array([bias]))

# Flatten kernel and dummy vector x (used by simulator to compute MVM)
W_flat = kernel.flatten()
np.save("W.npy", W_flat)
np.save("x.npy", np.zeros((9,)))  # dummy, replaced in simulator

# Start ISA generation
isa_lines = []
isa_lines.append("LOAD 2 200")  # Load kernel into R2 (assume kernel at addr 200)

output_index = 0

for i in range(output_size):
    for j in range(output_size):
        patch_base_addr = 100 + output_index * 10
        output_addr = 1000 + output_index * 10

        isa_lines.append(f"# Patch ({i},{j})")
        isa_lines.append(f"SC_ADDI 1 0 {patch_base_addr}")  # Set patch base address in R1
        isa_lines.append(f"CIM_MVM 1 2 3")  # R3 = W · X
        isa_lines.append(f"SC_ADDI 4 3 {bias}")  # R4 = R3 + bias
        isa_lines.append("RELU 5 4")  # R5 = max(0, R4)
        isa_lines.append(f"STORE 5 {output_addr}")  # Save R5 to output location

        output_index += 1

# Save ISA to file
with open("program.isa", "w") as f:
    f.write("\n".join(isa_lines))

print("✅ CNN ISA generation complete.")
