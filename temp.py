import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_copl_diagram():
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    
    # ==========================================
    # Panel 1: Graph Construction (Disjoint but Connected)
    # ==========================================
    ax[0].set_title("(a) Graph Construction with Item-Item Similarity", fontsize=14, pad=20)
    ax[0].set_xlim(0, 10)
    ax[0].set_ylim(0, 10)
    ax[0].axis('off')
    
    # User Nodes
    users = {'u1': (2, 7.5), 'u2': (2, 2.5)}
    # Item Nodes (Disjoint sets for u1 and u2)
    # u1's items
    items_u1 = {'i1': (5, 8.5), 'i2': (5, 6.5)}
    # u2's items
    items_u2 = {'i3': (5, 3.5), 'i4': (5, 1.5)}
    
    all_items = {**items_u1, **items_u2}
    
    # Draw Nodes
    for name, pos in users.items():
        circle = patches.Circle(pos, 0.4, facecolor='#333333', edgecolor='black', zorder=10)
        ax[0].add_patch(circle)
        ax[0].text(pos[0], pos[1], name, color='white', ha='center', va='center', fontweight='bold', fontsize=12, zorder=11)
        ax[0].text(pos[0]-0.8, pos[1], "User", ha='center', va='center', fontsize=10, color='gray')

    for name, pos in all_items.items():
        rect = patches.FancyBboxPatch((pos[0]-0.4, pos[1]-0.3), 0.8, 0.6, boxstyle="round,pad=0.1", 
                                      facecolor='#ffffff', edgecolor='black', zorder=10)
        ax[0].add_patch(rect)
        ax[0].text(pos[0], pos[1], name, color='black', ha='center', va='center', fontsize=12, zorder=11)
        # Dummy waveform
        x_wave = np.linspace(pos[0]-0.3, pos[0]+0.3, 20)
        y_wave = pos[1] - 0.2 + 0.1 * np.sin(4 * np.pi * (x_wave - pos[0]))
        ax[0].plot(x_wave, y_wave, color='gray', linewidth=0.5, zorder=11)

    # Draw User-Item Edges (Bipartite)
    # u1 -> i1 (Good/Blue), u1 -> i2 (Bad/Red)
    ax[0].plot([users['u1'][0], items_u1['i1'][0]], [users['u1'][1], items_u1['i1'][1]], color='#4285F4', linewidth=2, label='Pos (Good)')
    ax[0].plot([users['u1'][0], items_u1['i2'][0]], [users['u1'][1], items_u1['i2'][1]], color='#EA4335', linewidth=2, label='Neg (Bad)')
    
    # u2 -> i3 (Good/Blue), u2 -> i4 (Bad/Red)
    ax[0].plot([users['u2'][0], items_u2['i3'][0]], [users['u2'][1], items_u2['i3'][1]], color='#4285F4', linewidth=2)
    ax[0].plot([users['u2'][0], items_u2['i4'][0]], [users['u2'][1], items_u2['i4'][1]], color='#EA4335', linewidth=2)

    # Draw Item-Item Edges (Similarity) - THE KEY PART
    # i1 <-> i3 (Similar patterns)
    ax[0].plot([items_u1['i1'][0], items_u2['i3'][0]], [items_u1['i1'][1], items_u2['i3'][1]], 
               color='#34A853', linewidth=3, linestyle='--', zorder=5)
    ax[0].text(5.2, 6, "High Similarity\n(Feature-based)", ha='left', va='center', color='#34A853', fontweight='bold', fontsize=10)

    # i2 <-> i4 (Similar patterns)
    ax[0].plot([items_u1['i2'][0], items_u2['i4'][0]], [items_u1['i2'][1], items_u2['i4'][1]], 
               color='#34A853', linewidth=3, linestyle='--', zorder=5)
    
    # Legend
    ax[0].legend(loc='upper left', fontsize=10)
    ax[0].text(5, 9.5, "Sensor Data Items (Disjoint IDs)", ha='center', fontsize=12, fontweight='bold')
    
    # ==========================================
    # Panel 2: Message Passing Flow
    # ==========================================
    ax[1].set_title("(b) Collaborative Information Flow", fontsize=14, pad=20)
    ax[1].set_xlim(0, 10)
    ax[1].set_ylim(0, 10)
    ax[1].axis('off')

    # Draw nodes again for context
    # Only draw relevant path: u2 -> i3 -> i1 -> u1
    
    # Positions (slightly shifted for flow visualization)
    pos_u2 = (8, 2)
    pos_i3 = (5, 3)
    pos_i1 = (5, 7)
    pos_u1 = (2, 8)

    # Nodes
    ax[1].add_patch(patches.Circle(pos_u2, 0.4, facecolor='#333333', edgecolor='black'))
    ax[1].text(pos_u2[0], pos_u2[1], "u2", color='white', ha='center', va='center', fontweight='bold')

    ax[1].add_patch(patches.FancyBboxPatch((pos_i3[0]-0.4, pos_i3[1]-0.3), 0.8, 0.6, boxstyle="round,pad=0.1", facecolor='white', edgecolor='black'))
    ax[1].text(pos_i3[0], pos_i3[1], "i3", color='black', ha='center', va='center')
    
    ax[1].add_patch(patches.FancyBboxPatch((pos_i1[0]-0.4, pos_i1[1]-0.3), 0.8, 0.6, boxstyle="round,pad=0.1", facecolor='white', edgecolor='black'))
    ax[1].text(pos_i1[0], pos_i1[1], "i1", color='black', ha='center', va='center')

    ax[1].add_patch(patches.Circle(pos_u1, 0.4, facecolor='#333333', edgecolor='black'))
    ax[1].text(pos_u1[0], pos_u1[1], "u1", color='white', ha='center', va='center', fontweight='bold')

    # Flow Arrows
    # 1. u2 preference embedded into i3
    ax[1].annotate("", xy=pos_i3, xytext=pos_u2, arrowprops=dict(arrowstyle="->", color='#4285F4', lw=3))
    ax[1].text(6.8, 2.2, "1. Preference\nSignal", color='#4285F4', fontsize=10)

    # 2. Item-Item similarity flow (i3 -> i1)
    ax[1].annotate("", xy=(pos_i1[0], pos_i1[1]-0.4), xytext=(pos_i3[0], pos_i3[1]+0.4), 
                   arrowprops=dict(arrowstyle="->", color='#34A853', lw=4, ls='--'))
    ax[1].text(5.2, 5, "2. Feature Sharing\n(Item-Item Edge)", color='#34A853', fontweight='bold', fontsize=11, ha='left')

    # 3. Aggregated info to u1
    ax[1].annotate("", xy=(pos_u1[0]+0.4, pos_u1[1]), xytext=(pos_i1[0]-0.4, pos_i1[1]), 
                   arrowprops=dict(arrowstyle="->", color='#4285F4', lw=3))
    ax[1].text(3.5, 7.2, "3. Update u1\nwith u2's logic", color='black', fontsize=10)

    # Explanation Box
    text_str = (
        "Even though u1 and u2 never rated the same item,\n"
        "information flows because i1 and i3 are similar.\n\n"
        r"$h_{i1} \leftarrow h_{i1} + \lambda_{ii} A_{i1,i3} h_{i3}$" "\n"
        r"$e_{u1} \leftarrow e_{u1} + \alpha (W e_{i1})$"
    )
    ax[1].text(5, 0.5, text_str, ha='center', va='bottom', fontsize=11, 
               bbox=dict(boxstyle="round", facecolor="#f0f0f0", edgecolor="gray"))

    plt.tight_layout()
    plt.savefig('copl_variant_diagram.png', dpi=150)

if __name__ == "__main__":
    draw_copl_diagram()