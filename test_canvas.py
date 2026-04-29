import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Test canvas buffer access
fig, ax = plt.subplots()
ax.text(0.5, 0.5, 'Test', ha='center', va='center')

canvas = fig.canvas
canvas.draw()

try:
    buffer_rgba = canvas.buffer_rgba()
    print(f"Buffer RGBA type: {type(buffer_rgba)}")
    print(f"Buffer RGBA shape: {buffer_rgba.shape if hasattr(buffer_rgba, 'shape') else 'no shape'}")

    image_numpy = np.frombuffer(buffer_rgba, dtype='uint8')
    print(f"Image numpy shape: {image_numpy.shape}")

    height, width = canvas.get_width_height()
    print(f"Canvas size: {width}x{height}")

    reshaped = image_numpy.reshape(height, width, 4)
    print(f"Reshaped: {reshaped.shape}")

    rgb = reshaped[:,:,:3]
    print(f"RGB shape: {rgb.shape}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

plt.close()