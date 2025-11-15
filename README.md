# Bennett Chatbot (ai.py)

This project is a small chatbot implemented with NLTK and PyTorch. The script `ai.py` trains a simple neural network on hard-coded intents, saves `model.pth`, runs a few test queries and then starts an interactive prompt.

Prerequisites
- Windows (instructions use PowerShell)
- Python 3.8+ (3.10/3.11 recommended)

Quick start (PowerShell)

1. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

3. Install dependencies

Option A — Recommended (install CPU-only PyTorch first to avoid GPU wheels):

```powershell
# Install CPU-only PyTorch (Windows)
pip install --index-url https://download.pytorch.org/whl/cpu torch
# Then install other requirements
pip install -r requirements.txt
```

Option B — Try installing everything via requirements (may pick a GPU wheel if available):

```powershell
pip install -r requirements.txt
```

4. Run the chatbot

```powershell
python ai.py
```

5. Run the Streamlit UI

```powershell
streamlit run streamlit_app.py
```

Notes
- The first run will download some NLTK corpora (this is done automatically by `ai.py`).
- Training runs for several epochs and will print training/testing loss and accuracy. It saves the trained model to `model.pth`.
- If you run into PyTorch wheel problems on Windows, use the official PyTorch install page (https://pytorch.org/) to get the correct pip command for your CUDA/CPU configuration.

What I changed
- Fixed Python special-method typos (`__init__`, `__len__`, `__getitem__`) so the dataset and model classes work correctly.
- Replaced an invalid NLTK download token with `omw-1.4`.

If you want, I can try running the script in this workspace (will require installing packages). Say "please run it" and I'll proceed to install and execute it here.