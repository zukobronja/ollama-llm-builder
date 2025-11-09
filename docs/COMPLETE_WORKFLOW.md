# Complete SmolVLM2 to Ollama Workflow

This document provides the complete, tested workflow for converting HuggingFace models to Ollama.

---

## CRITICAL: Vision Quality Issue

**The Ollama-deployed SmolVLM2 has broken vision recognition due to:**
1. Missing multimodal projector (mmproj) file
2. Quantization severely degrading vision encoder
3. Lack of official Ollama support ([#9559](https://github.com/ollama/ollama/issues/9559))

**For vision tasks, use HuggingFace Transformers directly** (torch.bfloat16, no quantization).

See [SMOLVLM2_VISION_ISSUES.md](SMOLVLM2_VISION_ISSUES.md) for complete analysis and solutions.

---

## Successfully Tested (Text Generation)

- **Model**: SmolVLM2-2.2B-Instruct
- **Hardware**: Intel i9-13900HX, NVIDIA RTX 4070 8GB VRAM, 64GB RAM
- **Date**: November 2025
- **Note**: Vision quality degraded - use HuggingFace for image/video tasks

---

## Prerequisites

### 1. System Requirements

- **Python**: 3.9+
- **UV package manager**: For fast dependency management
- **llama.cpp**: For GGUF conversion and quantization
- **Ollama**: For model serving
- **CUDA**: Optional, for GPU acceleration

### 2. Install Dependencies

```fish
# Clone the repository
cd SmolVLM2

# Create virtual environment and install dependencies
uv sync

# Activate environment
source .venv/bin/activate
```

### 3. Set Up HuggingFace Token

Create a `.env` file:

```bash
HF_TOKEN=your_huggingface_token_here
```

---

## Step-by-Step Workflow

### Step 1: Download Model from HuggingFace

**Command:**
```fish
python scripts/download_model.py HuggingFaceTB/SmolVLM2-2.2B-Instruct --verbose
```

**Output:**
- Downloaded to: `models/source/SmolVLM2-2.2B-Instruct/`
- Size: ~8.37 GB
- Files: 2 safetensors, 11 config files, 2 tokenizer files

**Time:** ~2 minutes (depends on internet speed)

---

### Step 2: Convert to GGUF and Quantize

**Important:** Set CUDA library path for quantization:

```fish
# Fish shell
set -x LD_LIBRARY_PATH /usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH
```

Or for Bash/Zsh:
```bash
export LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH
```

**Find your CUDA library:**
```bash
find /usr -name "libcudart.so.12" 2>/dev/null
```

**Command:**
```fish
python scripts/conversion/convert_to_gguf.py \
 --model-id HuggingFaceTB/SmolVLM2-2.2B-Instruct \
 --source models/source/SmolVLM2-2.2B-Instruct \
 --quantize Q4_K_M,Q5_K_M,Q8_0 \
 --llama-cpp /mnt/data/projects/tools/llama.cpp \
 --verbose
```

**Output:**

| File | Size | Quality | VRAM | Use Case |
|------|------|---------|------|----------|
| SmolVLM2-2.2B-Instruct-F16.gguf | 3.4 GB | Best | ~6 GB | Reference |
| SmolVLM2-2.2B-Instruct-Q8_0.gguf | 1.8 GB | Very High | ~4 GB | High quality |
| SmolVLM2-2.2B-Instruct-Q5_K_M.gguf | 1.3 GB | Good | ~3 GB | Balanced |
| SmolVLM2-2.2B-Instruct-Q4_K_M.gguf | 1.1 GB | Good | ~2.8 GB | **8GB VRAM** |

**Time:**
- FP16 conversion: ~15 seconds
- Each quantization: ~30 seconds

---

### Step 3: Generate Ollama Modelfile

**Command:**
```fish
python scripts/generate_modelfile.py \
 --model-id HuggingFaceTB/SmolVLM2-2.2B-Instruct \
 --gguf models/gguf/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf \
 --verbose
```

**Output:**
- Created: `modelfiles/SmolVLM2-Q4_K_M.modelfile`
- Optimized for: laptop_rtx4070 profile
- Parameters: temperature 0.7, top_p 0.9, num_ctx 4096

**Modelfile contains:**
- GGUF file path
- Hardware-optimized parameters
- System prompt for vision-language tasks
- Template for vision-language interaction
- Stop tokens

---

### Step 4: Deploy to Ollama

**Important:** Use **tags** to keep multiple quantizations available!

**Create models with different tags:**
```fish
# Q4_K_M (recommended for 8GB VRAM)
ollama create smolvlm2:q4_k_m -f modelfiles/SmolVLM2-Q4_K_M.modelfile

# Q5_K_M (better quality)
ollama create smolvlm2:q5_k_m -f modelfiles/SmolVLM2-Q5_K_M.modelfile

# Q8_0 (best quality)
ollama create smolvlm2:q8_0 -f modelfiles/SmolVLM2-Q8_0.modelfile

# F16 (reference quality)
ollama create smolvlm2:f16 -f modelfiles/SmolVLM2-F16.modelfile
```

**Verify:**
```fish
ollama list
```

You should see:
```
NAME    ID    SIZE  MODIFIED
smolvlm2:q4_k_m  abc123   1.1 GB X minutes ago
smolvlm2:q5_k_m  def456   1.3 GB X minutes ago
smolvlm2:q8_0  ghi789   1.8 GB X minutes ago
smolvlm2:f16  jkl012   3.4 GB X minutes ago
```

**Set a default (optional):**
```fish
# Make Q4_K_M the default (tagged as 'latest')
ollama create smolvlm2:latest -f modelfiles/SmolVLM2-Q4_K_M.modelfile
```

---

### Step 5: Test the Models

**Test different quantizations:**

```fish
# Test Q4_K_M (fastest, lowest VRAM)
ollama run smolvlm2:q4_k_m "Hello, who are you?"

# Test Q5_K_M (better quality)
ollama run smolvlm2:q5_k_m "Hello, who are you?"

# Test Q8_0 (best quality)
ollama run smolvlm2:q8_0 "Hello, who are you?"

# Test F16 (reference quality)
ollama run smolvlm2:f16 "Hello, who are you?"

# Use default (if you set 'latest' tag)
ollama run smolvlm2 "Hello, who are you?"
```

**With image (if vision support works):**
```fish
ollama run smolvlm2:q4_k_m "Describe this image" --image path/to/image.jpg
```

**Compare quantizations:**
You can now easily compare quality vs performance across different quantizations!

---

## Complete Command Summary

Here's the entire workflow in one place:

```fish
# 1. Download model
python scripts/download_model.py HuggingFaceTB/SmolVLM2-2.2B-Instruct --verbose

# 2. Set CUDA library path
set -x LD_LIBRARY_PATH /usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH

# 3. Convert and quantize (creates all 4 versions)
python scripts/conversion/convert_to_gguf.py \
 --model-id HuggingFaceTB/SmolVLM2-2.2B-Instruct \
 --source models/source/SmolVLM2-2.2B-Instruct \
 --quantize Q4_K_M,Q5_K_M,Q8_0 \
 --llama-cpp /mnt/data/projects/tools/llama.cpp \
 --verbose

# 4. Generate Modelfiles for each quantization
python scripts/generate_modelfile.py \
 --model-id HuggingFaceTB/SmolVLM2-2.2B-Instruct \
 --gguf models/gguf/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf

python scripts/generate_modelfile.py \
 --model-id HuggingFaceTB/SmolVLM2-2.2B-Instruct \
 --gguf models/gguf/SmolVLM2-2.2B-Instruct-Q5_K_M.gguf

python scripts/generate_modelfile.py \
 --model-id HuggingFaceTB/SmolVLM2-2.2B-Instruct \
 --gguf models/gguf/SmolVLM2-2.2B-Instruct-Q8_0.gguf

# 5. Deploy all to Ollama with different tags
ollama create smolvlm2:q4_k_m -f modelfiles/SmolVLM2-Q4_K_M.modelfile
ollama create smolvlm2:q5_k_m -f modelfiles/SmolVLM2-Q5_K_M.modelfile
ollama create smolvlm2:q8_0 -f modelfiles/SmolVLM2-Q8_0.modelfile

# Optional: Set Q4_K_M as default
ollama create smolvlm2:latest -f modelfiles/SmolVLM2-Q4_K_M.modelfile

# 6. Test different versions
ollama run smolvlm2:q4_k_m "Hello, who are you?"
ollama run smolvlm2:q5_k_m "Hello, who are you?"
ollama run smolvlm2:q8_0 "Hello, who are you?"
```

---

## Files Created

```
SmolVLM2/
├── models/
│ ├── source/
│ │ └── SmolVLM2-2.2B-Instruct/  # Downloaded model (8.37 GB)
│ └── gguf/
│  ├── SmolVLM2-2.2B-Instruct-F16.gguf  # 3.4 GB
│  ├── SmolVLM2-2.2B-Instruct-Q4_K_M.gguf # 1.1 GB
│  ├── SmolVLM2-2.2B-Instruct-Q5_K_M.gguf # 1.3 GB
│  └── SmolVLM2-2.2B-Instruct-Q8_0.gguf  # 1.8 GB
└── modelfiles/
 └── SmolVLM2-Q4_K_M.modelfile  # Ollama configuration
```

**Total disk space used:** ~15 GB (source + all quantizations)

---

## Hardware Profile Optimization

The system automatically optimizes based on your hardware profile:

### Laptop RTX 4070 (8GB VRAM) - Default
- **Recommended**: Q4_K_M
- **VRAM usage**: ~2.8 GB
- **Quality**: Good
- **Speed**: Very Fast

### Workstation RTX 4090 (24GB VRAM)
- **Recommended**: Q8_0, Q6_K, Q5_K_M
- **VRAM usage**: 3-6 GB
- **Quality**: Very High to High
- **Speed**: Fast

### Server A100 (80GB VRAM)
- **Recommended**: F16, Q8_0
- **VRAM usage**: 4-8 GB
- **Quality**: Best
- **Speed**: Moderate to Fast

### CPU Only (No GPU)
- **Recommended**: Q4_K_M, Q4_0
- **RAM usage**: ~4 GB
- **Quality**: Good to Acceptable
- **Speed**: Slower (CPU inference)

---

## Troubleshooting

### Issue: "libcudart.so.12: cannot open shared object file"

**Solution:**
```fish
set -x LD_LIBRARY_PATH /usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH
```

Find your CUDA library:
```bash
find /usr -name "libcudart.so.12" 2>/dev/null
```

### Issue: "llama.cpp not found"

**Solution:**
Specify the path explicitly:
```fish
--llama-cpp /path/to/your/llama.cpp
```

### Issue: "No module named 'mistral_common'"

**Solution:**
```fish
uv sync # Reinstall dependencies
```

### Issue: Model conversion fails

**Solution:**
- Check if model architecture is supported by llama.cpp
- Ensure sufficient RAM (need ~16GB for 2.2B model)
- Try with a smaller model first (500M variant)

---

## What You Learned

1. **Download**: How to download models from HuggingFace with authentication
2. **Convert**: How to convert PyTorch/SafeTensors to GGUF format
3. **Quantize**: How to create multiple quantization levels for different hardware
4. **Configure**: How to create hardware-optimized Modelfiles
5. **Deploy**: How to deploy custom models to Ollama

---

## Next Steps

### Try Other Models

Use the same workflow for other models:

```fish
# SmolVLM2 500M (faster, smaller)
python scripts/download_model.py HuggingFaceTB/SmolVLM2-500M-Video-Instruct

# SmolVLM2 256M (smallest)
python scripts/download_model.py HuggingFaceTB/SmolVLM2-256M-Video-Instruct
```

### Create Different Quantizations

```fish
# Just Q4_K_M (fastest conversion)
--quantize Q4_K_M

# Best quality
--quantize Q8_0

# All options
--quantize Q4_K_M,Q5_K_M,Q6_K,Q8_0
```

### Use Different Hardware Profiles

```fish
# For workstation
python scripts/generate_modelfile.py \
 --gguf models/gguf/SmolVLM2-2.2B-Instruct-Q8_0.gguf \
 --profile workstation_rtx4090

# For CPU-only
python scripts/generate_modelfile.py \
 --gguf models/gguf/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf \
 --profile cpu_only
```

---

## Performance Benchmarks

Tested on: Intel i9-13900HX, RTX 4070 8GB, 64GB RAM

| Quantization | VRAM Usage | Tokens/sec | Quality |
|--------------|------------|------------|---------|
| Q4_K_M | ~2.8 GB | Fast | Good |
| Q5_K_M | ~3.2 GB | Fast | Good |
| Q8_0 | ~4.0 GB | Moderate | Very High |

*Note: Actual performance varies based on prompt length and context*

---

## Educational Value

This project demonstrates:

- **Model Format Conversion**: PyTorch → GGUF
- **Quantization Techniques**: FP16 → Q8/Q5/Q4
- **Hardware Optimization**: Profile-based configuration
- **Model Deployment**: Local AI serving with Ollama
- **Configuration Management**: YAML-based hardware profiles
- **Automation**: Python scripts for repeatable workflows

---

## Resources

- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [llama.cpp Documentation](https://github.com/ggerganov/llama.cpp)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [SmolVLM2 on HuggingFace](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)
- [Quantization Guide](docs/GGUF_CONVERSION.md)
