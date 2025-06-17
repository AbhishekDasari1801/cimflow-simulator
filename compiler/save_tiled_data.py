import numpy as np

# Large FC Layer
W = np.random.randint(0, 5, (1024, 64))
x = np.random.randint(0, 5, (64,))
b = np.random.randint(0, 10, (1024,))

np.save("W.npy", W)
np.save("x.npy", x)
np.save("b.npy", b)

print("✅ Tiled Wx + b data saved.")

