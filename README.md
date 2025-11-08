# Ollama LLM Builder

Build and deploy state-of-the-art LLMs to Ollama that are not currently available in the official model library.

## Quick Start

### Already Deployed? Run It!

```fish
ollama run smolvlm2 "Hello, introduce yourself"
```

### Build From Scratch

```bash
# 1. Install dependencies
uv sync
source .venv/bin/activate

# 2. Download model
python scripts/download_model.py HuggingFaceTB/SmolVLM2-2.2B-Instruct

# 3. Convert to GGUF (set CUDA path for Fish shell)
set -x LD_LIBRARY_PATH /usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH

python scripts/conversion/convert_to_gguf.py \
  --model-id HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --source models/source/SmolVLM2-2.2B-Instruct \
  --quantize Q4_K_M \
  --llama-cpp /path/to/llama.cpp

# 4. Generate Modelfile and deploy
python scripts/generate_modelfile.py \
  --model-id HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --gguf models/gguf/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf

ollama create smolvlm2 -f modelfiles/SmolVLM2-Q4_K_M.modelfile
```

See **[docs/COMPLETE_WORKFLOW.md](docs/COMPLETE_WORKFLOW.md)** for the full tested workflow.

## Models

### SmolVLM2-2.2B-Instruct - WORKING
- **Type**: Vision-Language Model
- **Status**: Successfully deployed to Ollama
- **HuggingFace**: [SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)
- **Size**: 3.4GB (FP16), 1.1GB (Q4_K_M recommended)
- **VRAM**: ~2.8GB (Q4_K_M on RTX 4070)
- **Quantizations Available**: F16, Q8_0, Q5_K_M, Q4_K_M

**Quick Test:**
```fish
# Use specific quantization with tags
ollama run smolvlm2:q4_k_m "Hello, who are you?"
ollama run smolvlm2:q5_k_m "Hello, who are you?"
ollama run smolvlm2:q8_0 "Hello, who are you?"
```

**Pro Tip:** Use tags to keep multiple quantizations:
```fish
ollama create smolvlm2:q4_k_m -f modelfiles/SmolVLM2-Q4_K_M.modelfile
ollama create smolvlm2:q5_k_m -f modelfiles/SmolVLM2-Q5_K_M.modelfile
```
See [docs/OLLAMA_TAGS.md](docs/OLLAMA_TAGS.md) for details.

## Hardware Profiles

This project is designed to work across different hardware configurations:

- **laptop_rtx4070** - i9-13900HX, RTX 4070 8GB VRAM (default)
- **workstation_rtx4090** - RTX 4090 24GB VRAM
- **server_a100** - A100 40GB/80GB VRAM
- **cpu_only** - CPU-only inference

Configure your hardware in `config/build_config.yaml`.

## Documentation

- **[docs/COMPLETE_WORKFLOW.md](docs/COMPLETE_WORKFLOW.md)** - **Complete tested workflow** (START HERE)
- **[docs/OLLAMA_TAGS.md](docs/OLLAMA_TAGS.md)** - How to use Ollama tags for multiple versions
- **[docs/SETUP.md](docs/SETUP.md)** - Installation and setup guide
- **[docs/GGUF_CONVERSION.md](docs/GGUF_CONVERSION.md)** - GGUF conversion educational guide
- **[docs/TASKS.md](docs/TASKS.md)** - Implementation tasks and planning

## Project Structure

```
ollama-llm-builder/
├── config/                # Hardware profiles and build configuration
├── docs/                  # Documentation
├── models/                # Downloaded/converted models
├── modelfiles/            # Ollama Modelfiles
├── scripts/               # Python scripts
│   ├── conversion/        # Model conversion scripts
│   └── utils/             # Utility scripts
├── tests/                 # Test scripts
├── build.py              # Main build orchestrator
└── pyproject.toml        # Project configuration
```

## Features

- **Configurable Hardware Profiles** - Optimized for different GPU/CPU configurations
- **Multiple Quantization Levels** - Q4, Q5, Q8, F16 support
- **Automated Pipeline** - Download → Convert → Quantize → Deploy
- **Compatibility Checking** - Verify hardware requirements before building
- **Vision Model Support** - Specialized support for vision-language models

## Usage

```bash
# List recommended models
python build.py --list-models

# Check compatibility
python build.py --model HuggingFaceTB/SmolVLM2-2.2B-Instruct --check-compatibility

# Download a model
python scripts/download_model.py HuggingFaceTB/SmolVLM2-2.2B-Instruct

# Build with default profile (laptop_rtx4070)
python build.py --model HuggingFaceTB/SmolVLM2-2.2B-Instruct

# Build with specific profile
python build.py --model HuggingFaceTB/SmolVLM2-500M-Video-Instruct --profile workstation_rtx4090

# Build with custom quantization
python build.py --model HuggingFaceTB/SmolVLM2-2.2B-Instruct --quantization Q5_K_M,Q8_0
```

## Status

**COMPLETE - Successfully Deployed SmolVLM2 to Ollama!**

- Project structure created
- Configuration system implemented
- Hardware profiles defined (4 profiles: laptop, workstation, server, CPU-only)
- Build orchestrator created
- Model download scripts (HuggingFace integration)
- GGUF conversion pipeline (FP16 + quantization)
- Modelfile generation (hardware-optimized)
- Successfully deployed to Ollama
- Complete workflow documented

**Tested Models:**
- SmolVLM2-2.2B-Instruct (4 quantizations: F16, Q8_0, Q5_K_M, Q4_K_M)
