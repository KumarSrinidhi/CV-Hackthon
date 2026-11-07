"""Entry point for the P4 depth-reconstruction app (HF Spaces + AWS App Runner).

HF's Gradio SDK runs this file and looks for a Gradio ``demo`` object. Building the
app at module import keeps the pipeline warm across requests. Locally / on App
Runner, ``demo.launch()`` binds to ``0.0.0.0:$PORT`` (default 8080).
"""

import os

from app.ui import build_app

demo = build_app()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "8080")),
    )
