# Documentation Index

## Quick Navigation

### Getting Started
- **[SETUP.md](SETUP.md)** - Installation and environment setup
- **[COMPLETE_WORKFLOW.md](COMPLETE_WORKFLOW.md)** - Full workflow for building models

### Technical Guides
- **[GGUF_CONVERSION.md](GGUF_CONVERSION.md)** - Educational guide on GGUF conversion
- **[OLLAMA_TAGS.md](OLLAMA_TAGS.md)** - Managing multiple quantizations with tags

### Important Issues
- **[SMOLVLM2_VISION_ISSUES.md](SMOLVLM2_VISION_ISSUES.md)** **CRITICAL** - Vision recognition broken in Ollama

### Task Planning
- **[TASKS.md](TASKS.md)** - Implementation roadmap and tasks

---

## Important Notice: Vision Models

**SmolVLM2 vision recognition is broken in Ollama deployments.**

### The Problem
- Missing multimodal projector (mmproj) file
- Quantization destroys vision encoder quality
- Ollama doesn't officially support SmolVLM2 yet

### The Solution
**Use HuggingFace Transformers directly for vision tasks:**
```python
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

# Load with bfloat16 (NO quantization)
model = AutoModelForImageTextToText.from_pretrained(
 "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
 torch_dtype=torch.bfloat16,
 device_map="cuda:0"
)
```

**Benefits:**
- Excellent vision quality
- ~6GB VRAM (fits RTX 4070 8GB)
- Video support
- No hallucinations

See [SMOLVLM2_VISION_ISSUES.md](SMOLVLM2_VISION_ISSUES.md) for complete details.

---

## Project Status

### Working Features
- Model download from HuggingFace
- GGUF conversion (F16, Q8_0, Q5_K_M, Q4_K_M)
- Modelfile generation
- Ollama deployment
- Hardware profile optimization
- Multiple quantization management (tags)

### Known Issues
- Vision recognition broken (SmolVLM2)
- No mmproj support in conversion
- Ollama lacks official SmolVLM2 support

### Best For
- Text-only models (works perfectly)
- SmolVLM2 text generation (limited usefulness)
- Learning GGUF conversion process
- Understanding quantization trade-offs

### Not Recommended For
- SmolVLM2 vision tasks (use HuggingFace instead)
- Production vision-language applications

---

## Documentation Quality

All documentation in this project is:
- Based on actual testing
- Hardware-specific (RTX 4070 8GB VRAM)
- Includes known issues and limitations
- Provides working alternatives

## Contributing

If you find issues or have improvements:
1. Test thoroughly on your hardware
2. Document the issue with evidence
3. Provide working alternatives when possible
4. Update relevant documentation files

---

## File Descriptions

### SETUP.md
Installation instructions, dependencies, hardware profiles configuration.

### COMPLETE_WORKFLOW.md
Step-by-step tested workflow from download to deployment. Includes warnings about vision quality.

### GGUF_CONVERSION.md
Educational guide explaining:
- What GGUF format is
- How conversion works
- Quantization levels and trade-offs
- When to use which quantization

### OLLAMA_TAGS.md
How to manage multiple quantizations:
- Using tags effectively
- Comparing quality across quantizations
- Storage management
- Best practices

### SMOLVLM2_VISION_ISSUES.md 
**Critical documentation** explaining:
- Why vision is broken in Ollama
- Technical root causes
- Working HuggingFace alternative
- Evidence and testing results
- Solutions and workarounds

### TASKS.md
Original implementation planning document:
- Phase-by-phase breakdown
- Research findings
- Blockers and risks
- Resources and references

---

## Quick Links

### External Resources
- [Ollama Documentation](https://github.com/ollama/ollama)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [SmolVLM2 on HuggingFace](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)
- [GGUF Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

### GitHub Issues
- [Ollama SmolVLM Support #9559](https://github.com/ollama/ollama/issues/9559)
- [Ollama Vision Architecture #7912](https://github.com/ollama/ollama/issues/7912)

### Learning Resources
- [SmolVLM Blog Post](https://huggingface.co/blog/smolvlm)
- [llama.cpp Vision Support](https://simonwillison.net/2025/May/10/llama-cpp-vision/)

---

**Last Updated**: November 2025
