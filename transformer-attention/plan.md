# Transformer Attention Mechanism - Manim Animation Plan

## Narrative Arc
**Misconception corrected**: "Attention is just a buzzword" → "Attention is a differentiable lookup table that learns which parts of the input to focus on."

**Aha Moment**: Seeing the softmax turn raw similarity scores into a probability distribution that dynamically weights the Value vectors. The geometric intuition of queries "attending to" keys in high-dimensional space.

**Visual Story**:
1. From RNN struggles (fading memory) to attention as a solution.
2. The core idea: "relevance scoring" + "weighted average".
3. Break down Q, K, V vectors with clear geometric animation.
4. Step-by-step matrix calculation with live visualization.
5. The "magic" of softmax turning scores into attention weights.
6. Multi-head as parallel attention subspaces.
7. Real example: attention map on a sentence (e.g. "The animal didn't cross the street because it was too tired").

**Target Length**: 4-6 minutes final video.

## Scene Breakdown

**Scene 1: The Problem with Sequences (0:00-0:45)**
- RNN/LSTM memory fading animation (gradient vanishing visually)
- Introduction of "Attention" as the hero
- Color: Warm academic palette

**Scene 2: The Big Idea - Weighted Attention (0:45-1:30)**
- Simple example: "Which word matters for 'it' in a sentence?"
- Animation of attention as arrows with varying thickness/opacity
- Formula introduction: Attention(Q, K, V)

**Scene 3: Query, Key, Value Explained (1:30-2:30)**
- 3D vector space visualization
- Query vector "searching" for matching Key vectors
- Dot product as similarity measure (cosine intuition)
- Classic 3B1B blue for Q, green for K, yellow for V

**Scene 4: Scaled Dot-Product Attention Formula (2:30-4:00)**
- Step by step matrix operations:
  1. Q @ K.T (similarity matrix)
  2. Scale by 1/sqrt(d_k) (why? prevent vanishing gradients - small animation)
  3. Softmax row-wise (beautiful probability distribution animation with curves)
  4. Multiply by V → context-aware output
- Live numbers updating on matrices
- Highlight numerical stability with scaling

**Scene 5: Multi-Head Attention (4:00-5:00)**
- Multiple heads running in parallel (split, compute, concat, linear)
- Show how different heads can attend to different patterns (syntax vs semantics)
- Final projection

**Scene 6: Attention in Action (5:00-6:00)**
- Real sentence example with attention heatmap animation
- "it" attending strongly to "animal"
- Transition to full Transformer diagram (high level)

**Scene 7: Conclusion & "Why It Works" (6:00-end)**
- Recap of differentiability + end-to-end learning
- Connection to modern LLMs
- Final beautiful visualization of attention patterns

## Visual Language
- **Background**: #1C1C1C (Classic 3B1B)
- **Primary**: #58C4DD (blue)
- **Secondary**: #83C167 (green)
- **Accent**: #FFFF00 (yellow)
- **Font**: Menlo/Monospace everywhere
- **Opacity**: Primary elements 1.0, supporting 0.4, grids 0.15
- **Pacing**: Generous `self.wait(1.5-3.0)` after every major reveal
- **Subcaptions**: Every `self.play()` has descriptive subcaption

## Technical Notes
- Use `MathTex` with raw strings
- Use `Group` for mixed Text + VMobject
- All scenes set `self.camera.background_color`
- Clean FadeOut at end of each scene
- High quality final render `-qh --format=mp4`

## Next Steps
1. Implement `script.py` with one class per scene
2. Render with `-ql` for iteration
3. Stitch scenes with ffmpeg
4. Add optional narration with TTS

This plan ensures pedagogical clarity, visual beauty, and technical accuracy.
