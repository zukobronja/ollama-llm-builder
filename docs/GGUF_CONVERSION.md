# GGUF Conversion Guide

## What is GGUF?

**GGUF** (GPT-Generated Unified Format) is a file format for storing models for inference with llama.cpp and Ollama. It's designed to be:

- **Efficient**: Optimized for fast loading and inference
- **Portable**: Works across different platforms
- **Quantizable**: Supports various quantization levels (Q4, Q5, Q8, etc.)
- **Self-contained**: Includes model weights, architecture, and tokenizer in one file

## Conversion Process Overview

```
HuggingFace Model (PyTorch/SafeTensors)
 ↓
[Step 1] Convert to GGUF (FP16)
 ↓
[Step 2] Quantize to smaller formats (Q4, Q5, Q8)
 ↓
GGUF Model ready for Ollama
```

## Step 1: Convert HuggingFace to GGUF (FP16)

### For Text-Only Models

llama.cpp provides `convert_hf_to_gguf.py`:

```bash
python llama.cpp/convert_hf_to_gguf.py \
 models/source/ModelName \
 --outtype f16 \
 --outfile models/gguf/ModelName-F16.gguf
```

**Parameters:**
- `--outtype f16`: Output in FP16 precision (full quality)
- `--outfile`: Where to save the GGUF file

### For Vision Models (Like SmolVLM2)

Vision models are more complex because they have:
1. **Vision Encoder** (processes images)
2. **Language Model** (generates text)
3. **Projector** (connects vision to language)

The conversion process:

```bash
# Check if your llama.cpp version supports SmolVLM2
python llama.cpp/convert_hf_to_gguf.py \
 models/source/SmolVLM2-2.2B-Instruct \
 --outtype f16 \
 --outfile models/gguf/SmolVLM2-2.2B-Instruct-F16.gguf
```

**Important:** Not all vision architectures are supported yet. SmolVLM2 support was added in llama.cpp in April 2025.

## Step 2: Quantization

Quantization reduces model size by lowering precision:

| Format | Precision | Size | Quality | Use Case |
|--------|-----------|------|---------|----------|
| F16 | 16-bit | 100% | Best | Reference/Baseline |
| Q8_0 | 8-bit  | 50% | Very High | High-end GPUs |
| Q6_K | 6-bit  | 40% | High | Good balance |
| Q5_K_M | 5-bit  | 35% | Good | Recommended |
| Q4_K_M | 4-bit  | 25% | Good | 8GB VRAM (your RTX 4070) |
| Q4_0 | 4-bit  | 25% | Acceptable | CPU-only |

### How to Quantize

**Using the conversion script (recommended):**

```fish
# Fish shell - Set CUDA library path first
set -x LD_LIBRARY_PATH /usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH

python scripts/conversion/convert_to_gguf.py \
 --model-id HuggingFaceTB/SmolVLM2-2.2B-Instruct \
 --source models/source/SmolVLM2-2.2B-Instruct \
 --quantize Q4_K_M,Q5_K_M,Q8_0 \
 --llama-cpp /path/to/llama.cpp \
 --verbose
```

```bash
# Bash/Zsh - Set CUDA library path first
export LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH

python scripts/conversion/convert_to_gguf.py \
 --model-id HuggingFaceTB/SmolVLM2-2.2B-Instruct \
 --source models/source/SmolVLM2-2.2B-Instruct \
 --quantize Q4_K_M,Q5_K_M,Q8_0 \
 --llama-cpp /path/to/llama.cpp \
 --verbose
```

**Manual quantization with llama-quantize:**

```bash
# Quantize to Q4_K_M (recommended for 8GB VRAM)
llama.cpp/llama-quantize \
 models/gguf/SmolVLM2-2.2B-Instruct-F16.gguf \
 models/gguf/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf \
 Q4_K_M

# Quantize to Q5_K_M (better quality, more VRAM)
llama.cpp/llama-quantize \
 models/gguf/SmolVLM2-2.2B-Instruct-F16.gguf \
 models/gguf/SmolVLM2-2.2B-Instruct-Q5_K_M.gguf \
 Q5_K_M

# Quantize to Q8_0 (best quality)
llama.cpp/llama-quantize \
 models/gguf/SmolVLM2-2.2B-Instruct-F16.gguf \
 models/gguf/SmolVLM2-2.2B-Instruct-Q8_0.gguf \
 Q8_0
```

## Step 3: Use with Ollama

Create a Modelfile:

```modelfile
FROM ./models/gguf/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf

TEMPLATE """{{- if .System }}System: {{ .System }}
{{ end }}User: {{ .Prompt }}
Assistant:"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096

SYSTEM "You are SmolVLM2, a vision-language AI assistant."
```

Then create the Ollama model:

```bash
ollama create smolvlm2 -f Modelfile
ollama run smolvlm2
```

## Understanding Quantization Methods

### K-quants (Recommended)

- **Q4_K_M**: 4-bit with medium quality k-quant
 - Best for VRAM-constrained systems (8GB)
 - ~25% of original size
 - Minimal quality loss

- **Q5_K_M**: 5-bit with medium quality k-quant
 - Better quality than Q4
 - ~35% of original size
 - Good balance

- **Q6_K**: 6-bit k-quant
 - Very high quality
 - ~40% of original size
 - For systems with more VRAM

### Legacy Quants

- **Q4_0**: Original 4-bit quantization
 - Fastest CPU inference
 - Lower quality than Q4_K_M

- **Q8_0**: 8-bit quantization
 - Near-original quality
 - ~50% size reduction

## Important: CUDA Library Path

If `llama-quantize` was built with CUDA support, you need to set the library path before running quantization.

**For Fish shell:**
```fish
set -x LD_LIBRARY_PATH /usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH
```

**For Bash/Zsh:**
```bash
export LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH
```

**Common CUDA library locations:**
- `/usr/local/lib/ollama/cuda_v12/` (Ollama's CUDA)
- `/usr/local/cuda/lib64/`
- `/usr/lib/x86_64-linux-gnu/`

**To find your CUDA library:**
```bash
find /usr -name "libcudart.so.12" 2>/dev/null
```

## Common Issues

### 1. "Unsupported architecture"

**Problem:** Your model architecture isn't supported by llama.cpp yet.

**Solution:**
- Check llama.cpp releases for support updates
- Use pre-converted GGUF models from HuggingFace
- Wait for support to be added

### 2. "Out of memory during conversion"

**Problem:** Not enough RAM to convert large models.

**Solution:**
- Close other applications
- Use swap space
- Convert on a machine with more RAM

### 3. "Quantization fails"

**Problem:** Invalid quantization type or corrupted F16 file.

**Solution:**
- Verify F16 file integrity
- Check `llama-quantize --help` for supported types
- Re-convert from HuggingFace if needed

### 4. "error while loading shared libraries: libcudart.so.12"

**Problem:** CUDA library not found by llama-quantize.

**Solution:**
Set the `LD_LIBRARY_PATH` environment variable:

```fish
# Fish shell
set -x LD_LIBRARY_PATH /usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH
```

```bash
# Bash/Zsh
export LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH
```

Find your CUDA library location:
```bash
find /usr -name "libcudart.so.12" 2>/dev/null
```

## Checking Conversion Status

### Verify GGUF file

```bash
# Use llama.cpp tools to inspect GGUF
llama.cpp/llama-cli \
 --model models/gguf/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf \
 --help

# Check file size
ls -lh models/gguf/*.gguf
```

### Expected Sizes for SmolVLM2-2.2B

- **F16**: ~4.4 GB
- **Q8_0**: ~2.2 GB
- **Q6_K**: ~1.7 GB
- **Q5_K_M**: ~1.5 GB
- **Q4_K_M**: ~1.1 GB

## Resources

- [llama.cpp Documentation](https://github.com/ggerganov/llama.cpp)
- [GGUF Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [Quantization Guide](https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md)
- [Model Support Matrix](https://github.com/ggerganov/llama.cpp#supported-models)

## Next Steps

After conversion, you can:
1. Test with llama.cpp directly
2. Create an Ollama Modelfile
3. Deploy to Ollama
4. Run inference tests
