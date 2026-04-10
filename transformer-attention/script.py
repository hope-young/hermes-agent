from manim import *
import numpy as np

# Classic 3B1B color palette
BG_COLOR = "#1C1C1C"
PRIMARY = "#58C4DD"    # Blue - Queries
SECONDARY = "#83C167"  # Green - Keys  
ACCENT = "#FFFF00"     # Yellow - Values
TEXT_COLOR = "#FFFFFF"
MONO = "Menlo"

config.background_color = BG_COLOR

class AttentionIntro(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        
        title = Text("Transformer Attention Mechanism", font_size=48, color=PRIMARY, font=MONO, weight=BOLD)
        subtitle = Text("From Intuition to Mathematics", font_size=32, color=SECONDARY, font=MONO)
        
        subtitle.next_to(title, DOWN, buff=0.8)
        
        self.play(Write(title), run_time=1.5)
        self.play(Write(subtitle), run_time=1.0)
        self.wait(2.0)
        
        self.play(FadeOut(Group(title, subtitle)), run_time=0.8)
        self.wait(0.5)


class QKVExplanation(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        self.add_subcaption("Query, Key, Value - The core of attention", duration=4)
        
        # Title
        title = Text("Q, K, V Vectors", font_size=42, color=PRIMARY, font=MONO, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Create vectors
        query = Arrow(ORIGIN, RIGHT*3, color=PRIMARY, buff=0, stroke_width=8)
        key = Arrow(ORIGIN, RIGHT*3, color=SECONDARY, buff=0, stroke_width=8).shift(DOWN*1.5)
        value = Arrow(ORIGIN, RIGHT*3, color=ACCENT, buff=0, stroke_width=8).shift(DOWN*3)
        
        q_label = Text("Query (what we're looking for)", font_size=24, color=PRIMARY, font=MONO).next_to(query, LEFT)
        k_label = Text("Key (what we compare against)", font_size=24, color=SECONDARY, font=MONO).next_to(key, LEFT)
        v_label = Text("Value (the information to retrieve)", font_size=24, color=ACCENT, font=MONO).next_to(value, LEFT)
        
        self.play(
            GrowArrow(query),
            Write(q_label),
            run_time=1.5
        )
        self.wait(0.8)
        
        self.play(
            GrowArrow(key),
            Write(k_label),
            run_time=1.5
        )
        self.wait(0.8)
        
        self.play(
            GrowArrow(value),
            Write(v_label),
            run_time=1.5
        )
        self.wait(2.0)
        
        # Similarity
        similarity = MathTex(r"\text{Similarity} = Q \cdot K", font_size=36, color=TEXT_COLOR)
        similarity.next_to(value, DOWN, buff=1.0)
        self.play(Write(similarity), run_time=1.5)
        self.wait(2.0)
        
        self.play(FadeOut(Group(*self.mobjects)), run_time=1.0)


class ScaledDotProduct(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        self.add_subcaption("Scaled Dot-Product Attention Formula", duration=6)
        
        title = Text("Scaled Dot-Product Attention", font_size=40, color=PRIMARY, font=MONO, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))
        
        # The formula
        formula = MathTex(
            r"\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V",
            font_size=36,
            color=TEXT_COLOR
        )
        formula.next_to(title, DOWN, buff=1.0)
        
        self.play(Write(formula), run_time=2.0)
        self.wait(1.5)
        
        # Break it down
        step1 = MathTex(r"1.\ QK^T\ \rightarrow\ \text{similarity scores}", font_size=28, color=SECONDARY)
        step2 = MathTex(r"2.\ \text{Scale by }\frac{1}{\sqrt{d_k}}\ \text{(numerical stability)}", font_size=28, color=SECONDARY)
        step3 = MathTex(r"3.\ \text{softmax}\ \rightarrow\ \text{probability distribution}", font_size=28, color=ACCENT)
        step4 = MathTex(r"4.\ \text{Weighted sum of Values}", font_size=28, color=PRIMARY)
        
        steps = VGroup(step1, step2, step3, step4).arrange(DOWN, aligned_edge=LEFT, buff=0.6)
        steps.next_to(formula, DOWN, buff=1.2)
        
        for i, step in enumerate(steps):
            self.play(Write(step), run_time=1.2)
            self.wait(0.8 if i < 3 else 2.0)
        
        self.wait(2.0)
        
        # Softmax animation hint
        softmax_title = Text("Softmax turns scores into weights", font_size=32, color=ACCENT, font=MONO)
        softmax_title.to_edge(DOWN)
        self.play(Write(softmax_title))
        self.wait(2.5)
        
        self.play(FadeOut(Group(*self.mobjects)), run_time=1.0)


class MultiHeadAttention(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        self.add_subcaption("Multiple attention heads learn different relationships", duration=4)
        
        title = Text("Multi-Head Attention", font_size=42, color=PRIMARY, font=MONO, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Visual of multiple heads
        heads = VGroup(*[
            Rectangle(width=1.2, height=2.0, color=BLUE, fill_opacity=0.3).shift(RIGHT*x*1.8)
            for x in range(-2, 3)
        ])
        heads.move_to(ORIGIN)
        
        head_labels = VGroup(*[
            Text(f"Head {i+1}", font_size=18, color=WHITE, font=MONO).next_to(rect, UP, buff=0.2)
            for i, rect in enumerate(heads)
        ])
        
        self.play(Create(heads), run_time=2.0)
        self.play(Write(head_labels), run_time=1.5)
        self.wait(1.0)
        
        concat = Text("Concatenate → Linear Projection", font_size=28, color=SECONDARY, font=MONO)
        concat.next_to(heads, DOWN, buff=1.5)
        arrow = Arrow(heads.get_bottom(), concat.get_top(), color=YELLOW)
        
        self.play(
            GrowArrow(arrow),
            Write(concat),
            run_time=1.8
        )
        self.wait(3.0)
        
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


class FinalScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        
        thank_you = Text("Thank You!", font_size=60, color=PRIMARY, font=MONO, weight=BOLD)
        subtitle = Text("Attention is all you need", font_size=36, color=SECONDARY, font=MONO)
        
        subtitle.next_to(thank_you, DOWN, buff=1.0)
        
        self.play(Write(thank_you), run_time=1.5)
        self.play(Write(subtitle), run_time=1.2)
        self.wait(4.0)
        
        self.play(FadeOut(Group(thank_you, subtitle)))


# To render specific scenes:
# manim -qh script.py AttentionIntro QKVExplanation ScaledDotProduct MultiHeadAttention FinalScene
