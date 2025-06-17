import numpy as np

# ---------------- Configuration ----------------
instr_latencies = {
    "SC_ADDI": 1,
    "MEM_CPY": 2,
    "LOAD": 1,
    "STORE": 1,
    "CIM_MVM": 25,
    "RELU": 1,
    "VEC_ADD": 5
}

instr_energy = {
    "SC_ADDI": 0.05,
    "MEM_CPY": 0.10,
    "LOAD": 0.20,
    "STORE": 0.20,
    "CIM_MVM": 2.5,
    "RELU": 0.05,
    "VEC_ADD": 0.50
}

total_cycles = 0
instr_counts = {}
instr_energy_used = {}
total_energy = 0.0  # picojoules

# ---------------- Instruction Loader ----------------
def load_instructions(filename):
    instructions = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0].startswith("#"):
                continue
            op = parts[0]
            if op == "SC_ADDI":
                instructions.append({"op": op, "rd": int(parts[1]), "rs": int(parts[2]), "imm": int(parts[3])})
            elif op == "MEM_CPY":
                instructions.append({"op": op, "src": int(parts[1]), "dst": int(parts[2])})
            elif op == "CIM_MVM":
                instructions.append({"op": op, "rs": int(parts[1]), "rt": int(parts[2]), "rd": int(parts[3])})
            elif op == "LOAD":
                instructions.append({"op": op, "rd": int(parts[1]), "addr": int(parts[2])})
            elif op == "STORE":
                instructions.append({"op": op, "rs": int(parts[1]), "addr": int(parts[2])})
            elif op == "VEC_ADD":
                instructions.append({"op": op, "rd": int(parts[1]), "rs": int(parts[2]), "rt": int(parts[3])})
            elif op == "RELU":
                instructions.append({"op": op, "rd": int(parts[1]), "rs": int(parts[2])})
    return instructions

# ---------------- Setup ----------------
instructions = load_instructions("program.isa")

# Allocate registers and memory
max_reg, max_mem = 0, 0
for instr in instructions:
    for key in ["rd", "rs", "rt"]:
        if key in instr:
            max_reg = max(max_reg, instr[key])
    if "addr" in instr:
        max_mem = max(max_mem, instr["addr"])

registers = [0] * (max_reg + 10)
memory = np.zeros((2000,), dtype=int)

# ---------------- CNN Patch + Kernel Loading ----------------
try:
    input_image = np.load("../compiler/input_image.npy")
    kernel = np.load("../compiler/kernel.npy")
    bias = int(np.load("../compiler/bias.npy")[0])
    weights = kernel.flatten().reshape(1, -1)  # shape (1, 9)

    # Load 6x6 = 36 patches into memory (starting from addr 100)
    kernel_size = 3
    output_size = 6
    for idx in range(output_size * output_size):
        i = idx // output_size
        j = idx % output_size
        patch = input_image[i:i+kernel_size, j:j+kernel_size].flatten()
        addr = 100 + idx * 10
        for k in range(len(patch)):
            memory[addr + k] = patch[k]

    print("✅ CNN patches loaded into memory.")

except Exception as e:
    print(f"⚠️  CNN inputs could not be loaded: {e}")
    exit()

# ---------------- Instruction Execution ----------------
for instr in instructions:
    op = instr["op"]
    instr_counts[op] = instr_counts.get(op, 0) + 1
    total_cycles += instr_latencies.get(op, 1)
    energy = instr_energy.get(op, 0.0)
    total_energy += energy
    instr_energy_used[op] = instr_energy_used.get(op, 0.0) + energy

    if op == "SC_ADDI":
        rd, rs, imm = instr["rd"], instr["rs"], instr["imm"]
        registers[rd] = registers[rs] + imm
        print(f"[SC_ADDI] R{rd} = R{rs}({registers[rs]}) + {imm} → {registers[rd]}")

    elif op == "MEM_CPY":
        src, dst = instr["src"], instr["dst"]
        memory[dst] = memory[src]
        print(f"[MEM_CPY] memory[{dst}] = memory[{src}]({memory[src]})")

    elif op == "LOAD":
        rd, addr = instr["rd"], instr["addr"]
        registers[rd] = memory[addr]
        print(f"[LOAD] R{rd} = memory[{addr}]({memory[addr]})")

    elif op == "STORE":
        rs, addr = instr["rs"], instr["addr"]
        memory[addr] = registers[rs]
        print(f"[STORE] memory[{addr}] = R{rs}({registers[rs]})")

    elif op == "CIM_MVM":
        rs, rt, rd = instr["rs"], instr["rt"], instr["rd"]
        patch_base = registers[rs]  # R1 holds address to patch
        mac_input = np.array([memory[patch_base + k] for k in range(9)])
        output = np.dot(weights, mac_input)
        registers[rd] = int(np.sum(output))
        print(f"[CIM_MVM] R{rd} = dot(kernel × patch) → {registers[rd]}")

    elif op == "VEC_ADD":
        rd, rs, rt = instr["rd"], instr["rs"], instr["rt"]
        registers[rd] = registers[rs] + registers[rt]
        print(f"[VEC_ADD] R{rd} = R{rs}({registers[rs]}) + R{rt}({registers[rt]}) → {registers[rd]}")

    elif op == "RELU":
        rd, rs = instr["rd"], instr["rs"]
        registers[rd] = max(0, registers[rs])
        print(f"[RELU] R{rd} = max(0, R{rs}({registers[rs]})) → {registers[rd]}")

# ---------------- Final Output ----------------
print("\nFinal Register States:", registers)

# Extract 6x6 output from memory and print
cnn_output = np.zeros((6, 6), dtype=int)
for i in range(6):
    for j in range(6):
        addr = 1000 + (i * 6 + j) * 10
        cnn_output[i, j] = memory[addr]

print("\n🧠 CNN Output Feature Map:")
print(cnn_output)
np.save("cnn_output.npy", cnn_output)

# Print Execution Summary
print("\nExecution Summary")
print("---------------------")
print(f"Total Instructions Executed: {len(instructions)}")
print(f"Total Cycles Taken         : {total_cycles}")
print("Instruction Usage Stats:")
for key, val in instr_counts.items():
    print(f"  - {key}: {val} times (×{instr_latencies[key]} cycles)")

print("\nEnergy Usage Report (approx., in picojoules)")
print("---------------------------------------------")
for op, energy in instr_energy_used.items():
    print(f"  - {op}: {energy:.2f} pJ ({instr_counts[op]} × {instr_energy[op]} pJ)")

print(f"\n⚡ Total Energy Consumed : {total_energy:.2f} pJ")
print(f"⚡ Average Energy/Instr  : {total_energy / len(instructions):.3f} pJ")

# Save final output (assume R5 has last output — not needed now but kept for consistency)
np.save("sim_output.npy", np.array([registers[5]]))
print(f"\n✅ Simulator output saved to sim_output.npy: {registers[5]}")

