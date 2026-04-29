import numpy as np
import imageio
import matplotlib.pyplot as plt

# Create a simple frame like the plotting function does
fig, ax = plt.subplots(figsize=(8, 6))
ax.text(0.5, 0.5, 'Test Frame', ha='center', va='center', fontsize=20)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

canvas = fig.canvas
canvas.draw()
image_numpy = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
image_numpy = image_numpy.reshape(*reversed(canvas.get_width_height()), 4)[:,:,:3]

plt.close()

print(f"Frame shape: {image_numpy.shape}")
print(f"Frame dtype: {image_numpy.dtype}")
print(f"Frame range: {image_numpy.min()} - {image_numpy.max()}")

# Test saving multiple frames
frames = [image_numpy, image_numpy]  # Duplicate frame for testing

try:
    imageio.mimsave('test_gif.gif', frames, fps=1, loop=1)
    print("GIF saved successfully")

    import os
    size = os.path.getsize('test_gif.gif')
    print(f"GIF size: {size} bytes")

except Exception as e:
    print(f"Error saving GIF: {e}")
    import traceback
    traceback.print_exc()