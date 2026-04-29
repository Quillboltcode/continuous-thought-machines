import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, ArrowStyle, ConnectionPatch

def create_full_ctm_flowchart():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Define positions for elements
    positions = {
        'input': (7, 9.5),
        'features': (7, 8.5),
        'init_state': (7, 7.5),
        'loop_start': (7, 6.5),
        'sync_action': (4, 5.5),
        'attention': (4, 4.5),
        'synapses': (4, 3.5),
        'memory': (4, 2.5),
        'nlm': (4, 1.5),
        'sync_output': (10, 5.5),
        'predictions': (10, 4.5),
        'loop_end': (7, 0.5),
        'output': (7, -0.5)
    }

    # Add text boxes
    def add_box(pos, text, width=3, height=0.8, color='lightblue'):
        rect = patches.FancyBboxPatch((pos[0]-width/2, pos[1]-height/2), width, height,
                                    boxstyle="round,pad=0.1", facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(pos[0], pos[1], text, ha='center', va='center', fontsize=9, wrap=True)

    def add_process(pos, text, width=2.8, height=0.7):
        add_box(pos, text, width, height, 'lightgreen')

    def add_decision(pos, text, width=2.8, height=0.7):
        add_box(pos, text, width, height, 'lightyellow')

    # Main components
    add_box(positions['input'], 'Input Data\n(images/text/etc.)', color='lightcyan')
    add_process(positions['features'], 'Feature Extraction\n(backbone + pos. emb.)')
    add_process(positions['init_state'], 'Initialize State\n(start_trace, start_activated)')
    add_decision(positions['loop_start'], 'For each thought iteration\n(t = 1 to T)')

    # Left branch (action/sync path)
    add_process(positions['sync_action'], 'Compute Action Sync\n(synchronisation_action)')
    add_process(positions['attention'], 'Cross-Attention\n(q from sync_action,\nkv from features)')
    add_process(positions['synapses'], 'Apply Synapses\n(recurrent mixing)')
    add_process(positions['memory'], 'Memory Update\n(FIFO or attention-based)')
    add_process(positions['nlm'], 'Neuron-Level Models\n(per-neuron processing)')

    # Right branch (output path)
    add_process(positions['sync_output'], 'Compute Output Sync\n(synchronisation_out)')
    add_process(positions['predictions'], 'Generate Predictions\n(output_projector)')

    add_decision(positions['loop_end'], 'Loop Complete?')
    add_box(positions['output'], 'Final Output\n(predictions, certainties,\nfinal_sync)', color='lightcyan')

    # Add arrows
    def add_arrow(start, end, label='', label_pos=None):
        arrow = FancyArrowPatch(start, end, arrowstyle=ArrowStyle('->', head_length=0.4, head_width=0.3),
                              color='black', linewidth=1.5, mutation_scale=15)
        ax.add_patch(arrow)
        if label:
            if label_pos:
                ax.text(label_pos[0], label_pos[1], label, ha='center', fontsize=8)
            else:
                mid_x, mid_y = (start[0] + end[0])/2, (start[1] + end[1])/2
                ax.text(mid_x, mid_y, label, ha='center', fontsize=8)

    # Main flow
    add_arrow((7, 9.1), (7, 8.9))  # input to features
    add_arrow((7, 8.1), (7, 7.9))  # features to init
    add_arrow((7, 7.1), (7, 6.9))  # init to loop start

    # Loop flow
    add_arrow((7, 6.1), (4, 5.9), 'action path')  # loop to sync_action
    add_arrow((4, 5.1), (4, 4.9))  # sync_action to attention
    add_arrow((4, 4.1), (4, 3.9))  # attention to synapses
    add_arrow((4, 3.1), (4, 2.9))  # synapses to memory
    add_arrow((4, 2.1), (4, 1.9))  # memory to nlm

    add_arrow((4, 1.1), (10, 5.9), 'output path')  # nlm to sync_output
    add_arrow((10, 5.1), (10, 4.9))  # sync_output to predictions

    add_arrow((10, 4.1), (7, 0.9), 'back to loop')  # predictions back to loop end
    add_arrow((7, 0.1), (7, -0.1))  # loop end to output

    # Loop back arrow
    add_arrow((7, 6.1), (7, 0.9), 'loop back', label_pos=(6, 3.5))

    # Add detailed component explanations
    explanations = [
        ('Synapses:', 'Mix information across neurons\n(U-Net or MLP)'),
        ('Memory:', 'Update trace history\n(FIFO sliding window)'),
        ('NLM:', 'Per-neuron MLPs on history'),
        ('Sync:', 'Pairwise neuron co-activations\nwith exponential decay'),
        ('Attention:', 'Internal state attends to\ninput features'),
    ]

    for i, (title, desc) in enumerate(explanations):
        ax.text(12, 9 - i*0.8, f'{title} {desc}', fontsize=8, ha='left')

    # Title
    ax.text(7, 9.8, 'CTM Full-Loop Reasoning Process Flowchart', ha='center', fontsize=16, fontweight='bold')

    # Legend
    ax.text(0.5, 9.5, 'Legend:', fontsize=10, fontweight='bold')
    ax.text(0.5, 9.0, 'Light Blue: I/O', fontsize=9)
    ax.text(0.5, 8.5, 'Light Green: Processing', fontsize=9)
    ax.text(0.5, 8.0, 'Light Yellow: Control Flow', fontsize=9)

    plt.tight_layout()
    plt.savefig('ctm_full_flowchart.png', dpi=150, bbox_inches='tight')
    plt.savefig('ctm_full_flowchart.pdf', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    create_full_ctm_flowchart()