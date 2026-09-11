import os
import sys
import subprocess

if __name__ == "__main__":
    app_path = os.path.join(os.path.dirname(__file__), "streamlit_app.py")
    port = os.getenv("STREAMLIT_PORT", "8501")

    print("=" * 65)
    print("🚀 Starting LLM API Optimizer - Streamlit Dashboard")
    print(f"🌐 Dashboard URL: http://localhost:{port}")
    print("=" * 65)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.port",
        port,
        "--server.headless",
        "true"
    ]
    subprocess.run(cmd)
