A minimal CNN for MNIST classification, with separate files for model, training and inference.

## Setup (Linux)
```bash
# Create venv
python3 -m venv myenv
source myenv/bin/activate
pip install -U pip

# Install PyTorch (choose ONE)
# GPU (CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# OR CPU only
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu~
