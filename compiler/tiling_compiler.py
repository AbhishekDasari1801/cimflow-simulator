import numpy as np

# Load input and parameters
W = np.load("W.npy")
x = np.load("x.npy")
b = np.load("b.npy")

tile_size = 256
tile_count = W.shape[0] // tile_size
x_addr = 100
bias = int(b[0])
repeat_block = [
    "REPEAT TILE_COUNT {",
    "  LOAD 2 TILE_ADDR",
    "  CIM_MVM 1 2 5",      # tile result to R5
    "  VEC_ADD 3 3 5",      # accumulate into R3
    "}"
]
def expand_repeat_with_accum(repeat_block, tile_count, tile_base=200):
    expanded = []
    expanded.append("CLEAR 3")  # Reset R3 before accumulation

    for i in range(tile_count):
        for line in repeat_block:
            if "REPEAT" in line or "{" in line or "}" in line:
                continue
            line = line.replace("TILE_ADDR", str(tile_base + i * 100))
            expanded.append(line)

    expanded.append(f"SC_ADDI 4 3 {bias}")   # Final bias
    expanded.append(f"RELU 5 4")     # R5 = RELU(R4)
    expanded.append(f"STORE 5 1000") # Save output

    return expanded
isa_lines = [f"LOAD 1 {x_addr}"]  # Input vector x
isa_lines += expand_repeat_with_accum(repeat_block, tile_count)
with open("program.isa", "w") as f:
    for line in isa_lines:
        f.write(line + "\n")

print("✅ Tiling + accumulation ISA generated.")

