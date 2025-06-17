# save_data.py
import numpy as np

x = np.random.randint(0, 5, (64,))
W = np.random.randint(0, 5, (512, 64))
b = 7  # or np.random.randint(0, 10)

np.save("input_x.npy", x)
np.save("weight_W.npy", W)
np.save("bias_b.npy", np.array([b]))

print("✅ Data saved as .npy files")

