import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, ArrowStyle

def create_sync_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Define positions for elements
    positions = {
        'input': (1, 5),
        'selection': (1, 4),
        'neurons': (1, 3),
        'products': (3, 3),
        'update': (5, 3),
        'output': (7, 3)
    }

    # Add text boxes
    def add_box(pos, text, width=2.5, height=0.8):
        rect = patches.FancyBboxPatch((pos[0]-width/2, pos[1]-height/2), width, height,
                                    boxstyle="round,pad=0.1", facecolor='lightblue', edgecolor='black')
        ax.add_patch(rect)
        ax.text(pos[0], pos[1], text, ha='center', va='center', fontsize=10, wrap=True)

    # Input
    add_box(positions['input'], 'Activated State\n(a^t)\n[d_model neurons]', height=1)

    # Selection
    add_box(positions['selection'], 'Neuron Selection\n(first-last/random/random-pairing)', height=0.8)

    # Neurons
    add_box(positions['neurons'], 'Left/Right\nNeuron Pairs', height=0.8)

    # Products
    add_box(positions['products'], 'Pairwise Products\nleft[i] × right[j]\n(or outer product)', width=3, height=1)

    # Update
    add_box(positions['update'], 'Recurrent Update\ns^t = (r×s^{t-1} + pairwise) /\n√(r×β^{t-1} + 1)', width=3, height=1.2)

    # Output
    add_box(positions['output'], 'Synchronization\nVector (s^t)', height=0.8)

    # Add arrows
    def add_arrow(start, end):
        arrow = FancyArrowPatch(start, end, arrowstyle=ArrowStyle('->', head_length=0.4, head_width=0.3),
                              color='black', linewidth=1.5, mutation_scale=15)
        ax.add_patch(arrow)

    add_arrow((1, 4.6), (1, 4.4))  # input to selection
    add_arrow((1, 3.6), (1, 3.4))  # selection to neurons
    add_arrow((1, 2.6), (2.5, 3))  # neurons to products
    add_arrow((3.5, 3), (4.5, 3))  # products to update
    add_arrow((6.5, 3), (6.5, 3))  # update to output

    # Add explanation text
    ax.text(5, 5.5, 'Synchronization Representation Computation', ha='center', fontsize=14, fontweight='bold')
    ax.text(5, 0.5, 'Creates compact representation of temporal neuron co-activation patterns', ha='center', fontsize=10, style='italic')

    # Add component explanations
    ax.text(8.5, 5, '• Neuron pairs chosen by selection type', fontsize=9, ha='left')
    ax.text(8.5, 4.5, '• Pairwise products capture co-activations', fontsize=9, ha='left')
    ax.text(8.5, 4, '• Exponential decay smooths over time', fontsize=9, ha='left')
    ax.text(8.5, 3.5, '• Used for attention queries & predictions', fontsize=9, ha='left')

    plt.tight_layout()
    plt.savefig('synchronization_diagram.png', dpi=150, bbox_inches='tight')
    plt.savefig('synchronization_diagram.pdf', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    create_sync_diagram()