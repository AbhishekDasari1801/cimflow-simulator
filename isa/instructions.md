# CIMFlow Instruction Set (v0.1)

## Format:
All instructions are 32 bits wide.
General layout: | Opcode (6) | RS (5) | RT (5) | RD (5) | Func/Imm (11) |

---

### Instruction Set:

1. **CIM_MVM RS, RT, RD**
   - Multiply [RS] with [RT] → store in [RD]
   - Used for in-memory matrix-vector multiplication.

2. **SC_ADDI RS, RT, Imm**
   - Scalar Add Immediate
   - RS = RT + Imm

3. **MEM_CPY RS, RT**
   - Copy memory from location in RS to location in RT

4. **VEC_ADD RS, RT, RD**
   - Vector Add: RD = RS + RT

5. **LOAD RD, Addr**
   - Load memory[Addr] → RD

6. **STORE RS, Addr**
   - Store RS → memory[Addr]

