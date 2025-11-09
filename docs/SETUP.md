# Setup Guide

## Installation with UV

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

### 1. Install UV (if not already installed)

```bash
# On Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

### 2. Create Virtual Environment and Install Dependencies

```bash
# Sync dependencies from pyproject.toml (creates venv automatically)
uv sync

# Or sync with development dependencies
uv sync --extra dev

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Alternative: Install in editable mode (older method)
# uv pip install -e .
```

### 3. Verify Installation

```bash
# Test the config manager
python scripts/utils/config_manager.py

# List available hardware profiles
python build.py --list-profiles

# List available models
python build.py --list-models
```

## Alternative: Traditional Installation

If you prefer using pip:

```bash
# Create virtual environment
python -m venv venv

# Activate
source venv/bin/activate # Linux/macOS
# or
venv\Scripts\activate # Windows

# Install from pyproject.toml
pip install -e .

# Or from requirements.txt
pip install -r requirements.txt
```

## GPU Support

### NVIDIA GPU (CUDA)

For NVIDIA GPUs with CUDA support, install PyTorch with CUDA:

```bash
# For CUDA 12.1 (adjust version as needed)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### AMD GPU (ROCm)

For AMD GPUs:

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
```

### CPU Only

If you don't have a GPU or want CPU-only:

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Ollama Installation

Ensure Ollama is installed on your system:

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS
brew install ollama

# Or download from https://ollama.com/download
```

Verify Ollama is running:

```bash
ollama --version
ollama list
```

## Configuration

### 1. Set Your Hardware Profile

Edit `config/build_config.yaml` and set `active_profile` to match your hardware:

```yaml
# For RTX 4070 laptop (default)
active_profile: "laptop_rtx4070"

# For RTX 4090 workstation
# active_profile: "workstation_rtx4090"

# For server with A100
# active_profile: "server_a100"

# For CPU-only
# active_profile: "cpu_only"
```

### 2. Review Hardware Profiles

Check available profiles and their recommendations:

```bash
python build.py --list-profiles
```

### 3. Check Compatibility

Before building, check if your hardware can run the model:

```bash
python build.py --model smolvlm2_instruct --check-compatibility
```

## Project Structure

After setup, your directory should look like:

```
SmolVLM2/
├── .venv/     # Virtual environment (created by uv)
├── config/    # Configuration files
│ ├── hardware_profiles.yaml
│ └── build_config.yaml
├── docs/     # Documentation
├── models/    # Downloaded/converted models (created during build)
├── modelfiles/   # Ollama Modelfiles (created during build)
├── scripts/    # Python scripts
│ ├── conversion/
│ └── utils/
├── tests/     # Test scripts
├── build.py    # Main build script
├── pyproject.toml  # Project configuration
└── requirements.txt  # Legacy requirements file
```

## Next Steps

1. **Test Configuration**:
 ```bash
 python build.py --list-profiles
 python build.py --check-compatibility --model smolvlm2_instruct
 ```

2. **Review Implementation Tasks**:
 ```bash
 cat docs/TASKS.md
 ```

3. **Start Building** (when scripts are implemented):
 ```bash
 python build.py --model smolvlm2_instruct
 ```

## Troubleshooting

### UV Not Found

Make sure UV is in your PATH:

```bash
export PATH="$HOME/.cargo/bin:$PATH" # Linux/macOS
```

### Import Errors

Make sure the virtual environment is activated and dependencies are installed:

```bash
source .venv/bin/activate
uv pip install -e .
```

### CUDA/GPU Not Detected

Verify your CUDA installation:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Ollama Not Found

Make sure Ollama is installed and in your PATH:

```bash
which ollama
ollama --version
```

## Hardware Requirements

### Minimum (for SmolVLM2 with Q4_K_M quantization)
- **RAM**: 16 GB
- **Storage**: 20 GB free space
- **VRAM**: 4 GB (or CPU-only)

### Recommended (RTX 4070 - default profile)
- **RAM**: 64 GB
- **Storage**: 50 GB free space (for multiple quantization variants)
- **VRAM**: 8 GB

### Optimal (for multiple models and high quality)
- **RAM**: 128+ GB
- **Storage**: 100+ GB SSD
- **VRAM**: 24+ GB
