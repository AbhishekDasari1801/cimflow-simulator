import numpy as np
import glob

isa = []
mem_addr = 100
weight_base = 200
store_base = 1000
input_reg = 1  # Input vector register

# Load input
isa.append(f"LOAD {input_reg} {mem_addr}")
mem_addr += 64  # Assume x is 64 long

# Detect and sort weight files
weight_files = sorted(glob.glob("W*.npy"))
bias_files = sorted(glob.glob("b*.npy"))

for i in range(len(weight_files)):
    w_file = weight_files[i]
    b_file = bias_files[i]
    layer = i + 1

    W = np.load(w_file)
    b = int(np.load(b_file)[0])

    weight_addr = weight_base + i * 100
    output_reg = input_reg + 2
    temp_reg = input_reg + 1
    store_addr = store_base + i * 100

    # LOAD weight
    isa.append(f"LOAD {temp_reg} {weight_addr}")

    # CIM MVM
    isa.append(f"CIM_MVM {input_reg} {temp_reg} {temp_reg+1}")

    # Add bias
    isa.append(f"SC_ADDI {output_reg} {temp_reg+1} {b}")

    # Store result
    isa.append(f"STORE {output_reg} {store_addr}")

    input_reg = output_reg  # Output becomes next input

# Write output
with open("program.isa", "w") as f:
    for line in isa:
        f.write(line + "\n")

print("✅ N-layer .isa program generated!")

