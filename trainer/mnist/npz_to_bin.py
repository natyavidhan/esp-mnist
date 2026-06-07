import numpy as np
import struct

def save_weights(filename, layers):
    """Save weights to binary file with headers"""
    with open(filename, 'wb') as f:
        # magic number — verify file not corrupt
        f.write(struct.pack('I', 0xDEADBEEF))
        # number of layers
        f.write(struct.pack('I', len(layers)))
        
        for w, b in layers:
            # weight shape info
            f.write(struct.pack('II', w.shape[0], w.shape[1]))
            # bias shape (flattened to 1D, store length)
            f.write(struct.pack('I', b.shape[0] if len(b.shape) == 1 else b.shape[0] * b.shape[1]))
            # raw arrays as float32
            w.astype(np.float32).tofile(f)
            b.astype(np.float32).tofile(f)

def load_weights(filename):
    """Load weights from binary file"""
    with open(filename, 'rb') as f:
        # read magic number
        magic = struct.unpack('I', f.read(4))[0]
        if magic != 0xDEADBEEF:
            raise ValueError("Invalid weights file: bad magic number")
        
        # read number of layers
        num_layers = struct.unpack('I', f.read(4))[0]
        
        layers = []
        for _ in range(num_layers):
            # read weight shape
            w_rows, w_cols = struct.unpack('II', f.read(8))
            # read bias shape
            b_len = struct.unpack('I', f.read(4))[0]
            
            # read arrays
            w = np.fromfile(f, dtype=np.float32, count=w_rows * w_cols).reshape((w_rows, w_cols))
            b = np.fromfile(f, dtype=np.float32, count=b_len).reshape((b_len, 1))
            
            layers.append((w, b))
    
    return layers

# Load the npz file
weights = np.load("mnist_weights.npz")

# Save using the binary format
layers = [
    (weights["W1"], weights["b1"]),
    (weights["W2"], weights["b2"]),
]
save_weights("mnist_weights.bin", layers)

print("Converted mnist_weights.npz -> mnist_weights.bin")
print(f"W1 shape: {weights['W1'].shape}")
print(f"b1 shape: {weights['b1'].shape}")
print(f"W2 shape: {weights['W2'].shape}")
print(f"b2 shape: {weights['b2'].shape}")

# Verify by loading it back
print("\nVerifying...")
loaded = load_weights("mnist_weights.bin")
print(f"Loaded {len(loaded)} layers")
for i, (w, b) in enumerate(loaded):
    print(f"  Layer {i}: W{w.shape}, b{b.shape}")
