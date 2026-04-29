import numpy as np
import imageio
import matplotlib.pyplot as plt

# Create a simple test frame
fig, ax = plt.subplots()
ax.text(0.5, 0.5, 'Test Frame', ha='center', va='center')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Convert to numpy array like the plotting function does
canvas = fig.canvas
canvas.draw()
image_numpy = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
image_numpy = (image_numpy.reshape(*reversed(canvas.get_width_height()), 4)[:,:,:3])

plt.close()

# Test saving
frames = [image_numpy]
try:
    imageio.mimsave('test.gif', frames, fps=1, loop=1)
    print("Test GIF created successfully")
except Exception as e:
    print(f"Error creating test GIF: {e}")