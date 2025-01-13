import os
import subprocess

def streamlit_run(dummy_input):
    streamlit_file = os.path.join(os.path.dirname(__file__), "streamlit.py")
    if not os.path.exists(streamlit_file):
        raise FileNotFoundError(f"File Not Found: {streamlit_file}")
    subprocess.run(["streamlit", "run", streamlit_file])