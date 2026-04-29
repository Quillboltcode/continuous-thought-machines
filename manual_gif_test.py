import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

# Simulate what the make_classification_gif function does
frames = []

# Create a few test frames
for i in range(3):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, f'Frame {i+1}', ha='center', va='center', fontsize=20)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    canvas = fig.canvas
    canvas.draw()
    image_numpy = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
    image_numpy = image_numpy.reshape(*reversed(canvas.get_width_height()), 4)[:,:,:3]

    frames.append(image_numpy)
    plt.close()

print(f"Created {len(frames)} frames")
print(f"Frame shapes: {[f.shape for f in frames]}")

# Try to save
try:
    imageio.mimsave('manual_test.gif', frames, fps=1, loop=1)
    print("Manual GIF saved successfully")
    import os
    print(f"Size: {os.path.getsize('manual_test.gif')} bytes")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()