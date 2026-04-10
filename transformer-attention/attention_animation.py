import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import os
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
plt.switch_backend('Agg')  # Non-interactive backend for faster rendering

# Set style
plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 10), facecolor='#1C1C1C')
gs = GridSpec(2, 3, figure=fig)

# Titles and labels
fig.suptitle('Transformer Attention Mechanism\n(Visualized with Matplotlib Animation)', fontsize=20, color='#58C4DD', fontfamily='monospace')

# Subplots
ax_qkv = fig.add_subplot(gs[0, 0])
ax_scores = fig.add_subplot(gs[0, 1])
ax_weights = fig.add_subplot(gs[0, 2])
ax_heatmap = fig.add_subplot(gs[1, :])

for ax in [ax_qkv, ax_scores, ax_weights, ax_heatmap]:
    ax.set_facecolor('#0F0F0F')
    for spine in ax.spines.values():
        spine.set_color('#58C4DD')

# Data
tokens = ['The', 'animal', 'did', "n't", 'cross', 'the', 'street', 'because', 'it', 'was', 'tired']
n = len(tokens)

# Dummy embeddings (for visualization)
np.random.seed(42)
Q = np.random.randn(n, 4)
K = np.random.randn(n, 4)
V = np.random.randn(n, 4)

def update(frame):
    ax_qkv.clear()
    ax_scores.clear()
    ax_weights.clear()
    ax_heatmap.clear()
    
    for ax in [ax_qkv, ax_scores, ax_weights, ax_heatmap]:
        ax.set_facecolor('#0F0F0F')
    
    progress = frame / 60.0  # 0 to ~4 seconds worth of animation
    
    # 1. QKV Visualization
    ax_qkv.set_title('Query, Key, Value Vectors', color='#58C4DD', fontsize=14, fontfamily='monospace')
    ax_qkv.set_xticks([])
    ax_qkv.set_yticks([])
    
    # Draw tokens
    for i, token in enumerate(tokens):
        color = '#58C4DD' if i == 8 else '#83C167'  # highlight "it"
        ax_qkv.text(0.1, 0.9 - i*0.08, token, color=color, fontsize=11, fontfamily='monospace')
    
    # Arrows for attention (from "it")
    if progress > 0.3:
        for i in range(n):
            alpha = 0.3 + 0.7 * np.exp(-((i-4)**2)/8) if i != 8 else 1.0
            ax_qkv.annotate('', xy=(0.6, 0.9 - 8*0.08), xytext=(0.3, 0.9 - i*0.08),
                           arrowprops=dict(arrowstyle='->', color='#FFFF00', alpha=alpha, lw=2))
    
    ax_qkv.text(0.7, 0.1, '"it" is querying\nother words', color='#FFFF00', fontsize=10, fontfamily='monospace')
    
    # 2. Similarity Scores (QK^T)
    ax_scores.set_title('Similarity Scores (Q · Kᵀ)', color='#83C167', fontsize=14, fontfamily='monospace')
    if progress > 0.4:
        scores = Q @ K.T
        scores = scores[8]  # for the word "it"
        im = ax_scores.imshow(scores.reshape(1, -1), cmap='viridis', aspect='auto')
        ax_scores.set_yticks([0])
        ax_scores.set_yticklabels(['"it"'])
        ax_scores.set_xticks(range(n))
        ax_scores.set_xticklabels(tokens, rotation=45, fontsize=9, color='white')
        plt.colorbar(im, ax=ax_scores, fraction=0.046, pad=0.04)
    
    # 3. Attention Weights (Softmax)
    ax_weights.set_title('Attention Weights = softmax(Scores / √d)', color='#FFFF00', fontsize=14, fontfamily='monospace')
    if progress > 0.6:
        scores = Q @ K.T
        scores = scores[8] / np.sqrt(4)
        weights = np.exp(scores) / np.sum(np.exp(scores))
        bars = ax_weights.bar(range(n), weights, color='#58C4DD')
        ax_weights.set_xticks(range(n))
        ax_weights.set_xticklabels(tokens, rotation=45, fontsize=9)
        ax_weights.set_ylabel('Weight', color='white')
        for bar, w in zip(bars, weights):
            height = bar.get_height()
            ax_weights.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{w:.2f}', ha='center', color='white', fontsize=9)
    
    # 4. Attention Heatmap
    ax_heatmap.set_title('Attention Map - How strongly "it" attends to each word', color='#58C4DD', fontsize=14, fontfamily='monospace')
    if progress > 0.7:
        attn_map = np.outer(np.exp(Q[8] @ K.T / 2), np.ones(4))  # simplified
        attn_map = attn_map / attn_map.max()
        im2 = ax_heatmap.imshow(attn_map, cmap='magma', aspect='auto')
        ax_heatmap.set_yticks([0])
        ax_heatmap.set_yticklabels(['"it"'])
        ax_heatmap.set_xticks(range(n))
        ax_heatmap.set_xticklabels(tokens, rotation=45, fontsize=10)
        plt.colorbar(im2, ax=ax_heatmap, fraction=0.046, pad=0.04, label='Attention Strength')
        
        # Highlight strongest
        strongest = np.argmax(Q[8] @ K.T)
        rect = patches.Rectangle((strongest-0.4, -0.4), 0.8, 1.8, linewidth=3, edgecolor='#FFFF00', facecolor='none')
        ax_heatmap.add_patch(rect)
        ax_heatmap.text(strongest, 0.6, 'Strongest\n("animal")', color='#FFFF00', ha='center', fontsize=11, fontweight='bold')
    
    # Progress indicator
    fig.text(0.02, 0.02, f'Frame {frame}/80   Progress: {progress:.1f}x', color='#666666', fontsize=10, fontfamily='monospace')
    
    return []

# Create animation
anim = FuncAnimation(fig, update, frames=80, interval=80, blit=False, repeat=True)

# Save to MP4
writer = FFMpegWriter(fps=15, metadata=dict(artist='Hermes Agent'), bitrate=1800)
output_path = "transformer_attention_visualization.mp4"
anim.save(output_path, writer=writer)

print(f"✅ MP4 generated successfully: {output_path}")
print(f"File size: {np.round(os.path.getsize(output_path)/1024/1024, 2)} MB")
