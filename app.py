"""Hugging Face Spaces entry point for the P4 depth-reconstruction app.

HF's Gradio SDK runs this file and looks for a Gradio ``demo`` object. Building the
app at module import keeps the pipeline warm across requests, and ``demo.launch()``
is the canonical Gradio-Spaces bootstrap.
"""

from app.ui import build_app

demo = build_app()

if __name__ == "__main__":
    demo.launch()
