# Ollama LLM Builder - Project Documentation

## Project Overview

This project implements and hosts state-of-the-art Large Language Models (LLMs) on Ollama that are not currently available in the official Ollama model library.

-----
NOTE: You are Principal Architect, Engineer, Developer and AI expert.
The gaol is to build LLMs for my Laptop primarly:
- CPU - i91390HX
- 64 GB RAM
- NVIDIA 4070 8GB VRAM
- 2SSD - 2 x 2TB

But make this configurable, that we can also build bigger models, if we run the code on buger machiens. I will have also bugger machines.
-----

## Purpose

Many cutting-edge open-source models from HuggingFace are not yet available on Ollama. This project bridges that gap by:

1. Converting models from HuggingFace/PyTorch format to Ollama-compatible format (GGUF)
2. Creating proper Modelfiles with optimized parameters
3. Testing and validating model performance
4. Documenting the implementation process for community use

## Current Implementation Status

### SmolVLM2 (In Progress)
- **Model Type**: Vision-Language Model (VLM)
- **Source**: HuggingFace Transformers
- **Status**: Planning phase
- **Complexity**: High (requires vision and language component integration)

## Project Structure

```
SmolVLM2/
├── README.md           # Project overview
├── CLAUDE.md          # Detailed documentation (this file)
├── docs/
│   └── TASKS.md       # Implementation tasks and planning
├── models/            # Converted model files (GGUF)
├── modelfiles/        # Ollama Modelfile definitions
├── scripts/           # Conversion and utility scripts
└── tests/             # Model testing scripts
```

## Key Challenges

### Vision-Language Models (VLMs)
Unlike text-only models, VLMs require:
- Separate vision encoder (for image processing)
- Language decoder (for text generation)
- Projection layers (to connect vision and language)
- Proper template formatting for multimodal inputs

### Conversion Process
1. Export HuggingFace model weights
2. Convert to GGUF format using llama.cpp tools
3. Create Modelfile with proper architecture parameters
4. Configure vision-specific settings
5. Test with sample images and prompts

## Dependencies

- Python 3.8+
- PyTorch
- HuggingFace Transformers
- Ollama
- llama.cpp (for GGUF conversion)
- sentencepiece/tokenizers

## Implementation Workflow

See `docs/TASKS.md` for detailed implementation tasks and progress tracking.

## Contributing

This is a learning project focused on understanding model conversion and deployment. Each implementation is documented thoroughly to help others learn the process.

## Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [llama.cpp GGUF Conversion](https://github.com/ggerganov/llama.cpp)
- [HuggingFace Model Hub](https://huggingface.co/models)
- [SmolVLM2 on HuggingFace](https://huggingface.co/HuggingFaceTB/SmolVLM2-Instruct)

## Notes

- This project is for educational purposes and experimentation
- Model conversion can be resource-intensive (requires significant RAM and storage)
- Not all models can be easily converted to GGUF format
- Vision models require Ollama version with multimodal support
