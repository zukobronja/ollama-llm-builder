# SmolVLM2 Implementation Tasks

## Project Goal
Implement and host SmolVLM2 (Vision-Language Model) on Ollama for local inference.

## Target Hardware
- **CPU**: Intel i9-13900HX
- **RAM**: 64 GB
- **GPU**: NVIDIA RTX 4070 (8GB VRAM) - Primary constraint
- **Storage**: 2 x 2TB SSD

## Hardware Optimization Strategy
Given the 8GB VRAM limitation, we will:
1. Use Q4_K_M quantization (4-bit) as primary target (~2-3GB VRAM)
2. Optimize `num_gpu_layers` to fit within VRAM budget
3. Leverage 64GB RAM for CPU offloading if needed
4. Create multiple quantization variants (Q4, Q5, Q8) for flexibility

---

## Phase 1: Research & Prerequisites

### 1.1 Model Research
- [ ] Research SmolVLM2 architecture on HuggingFace
  - Model size and variants (Instruct vs Base)
  - Architecture components (vision encoder, language decoder, projector)
  - Input/output format requirements
- [ ] Check Ollama's vision model support status
  - Minimum Ollama version required for VLMs
  - Supported vision architectures
  - Existing vision model examples (llava, bakllava)
- [ ] Document SmolVLM2 specifications
  - Parameter count
  - Context length
  - Image resolution requirements
  - Tokenizer type

### 1.2 Environment Setup
- [ ] Install required dependencies
  ```bash
  pip install torch transformers sentencepiece
  pip install huggingface_hub
  ```
- [ ] Clone/download llama.cpp for GGUF conversion
  ```bash
  git clone https://github.com/ggerganov/llama.cpp
  cd llama.cpp
  make
  ```
- [ ] Verify Ollama version supports vision models
  ```bash
  ollama --version
  ```
- [ ] Check available disk space (need ~20GB for model + conversion)

---

## Phase 2: Model Acquisition & Analysis

### 2.1 Download SmolVLM2
- [ ] Create Python script to download model from HuggingFace
  - Location: `scripts/download_smolvlm2.py`
  - Download model weights
  - Download tokenizer files
  - Download config.json
- [ ] Test model locally with HuggingFace Transformers
  - Verify model loads correctly
  - Test with sample image
  - Document input/output format

### 2.2 Model Architecture Analysis
- [ ] Analyze config.json structure
  - Identify vision encoder type (CLIP, SigLIP, etc.)
  - Identify language model backbone
  - Document projection layer specifications
- [ ] Map architecture to Ollama-compatible format
  - Check if similar to llava architecture
  - Identify any custom components
- [ ] Document findings in `docs/ARCHITECTURE.md`

---

## Phase 3: Model Conversion

### 3.1 GGUF Conversion Preparation
- [ ] Research conversion process for vision models
  - Check llama.cpp support for SmolVLM2 architecture
  - Find conversion scripts (e.g., `convert-hf-to-gguf.py`)
  - Study existing VLM conversion examples
- [ ] Prepare conversion script
  - Location: `scripts/convert_to_gguf.py`
  - Handle vision encoder separately if needed
  - Handle language model conversion
  - Handle projection layers

### 3.2 Execute Conversion
- [ ] Convert SmolVLM2 to GGUF format
  ```bash
  python scripts/convert_to_gguf.py \
    --model-dir ./models/smolvlm2-hf \
    --output-dir ./models/smolvlm2-gguf
  ```
- [ ] Verify GGUF file integrity
  - Check file size is reasonable
  - Validate with llama.cpp tools
- [ ] Create quantized versions (optional)
  - Q4_K_M (recommended for most users)
  - Q5_K_M (better quality)
  - Q8_0 (highest quality)

---

## Phase 4: Ollama Integration

### 4.1 Create Modelfile
- [ ] Create base Modelfile for SmolVLM2
  - Location: `modelfiles/SmolVLM2-Base.modelfile`
  - Reference the GGUF file path
  - Set vision-specific parameters
- [ ] Configure parameters
  ```
  FROM ./models/smolvlm2-gguf/model.gguf

  # Vision model parameters
  PARAMETER vision_encoder <encoder_type>
  PARAMETER image_size <size>

  # Generation parameters
  PARAMETER temperature 0.7
  PARAMETER top_p 0.9
  PARAMETER num_ctx 4096
  PARAMETER num_gpu_layers -1

  # System prompt
  SYSTEM "You are SmolVLM2, a vision-language model that can analyze images and answer questions about them."

  # Template for vision-language interaction
  TEMPLATE """[INST]<image>{{.Image}}</image>
  {{.Prompt}}[/INST]"""
  ```
- [ ] Research proper template format for SmolVLM2
  - Check HuggingFace model card for prompt format
  - Test different template variations

### 4.2 Build Ollama Model
- [ ] Create model in Ollama
  ```bash
  ollama create smolvlm2 -f modelfiles/SmolVLM2-Base.modelfile
  ```
- [ ] Troubleshoot any errors
  - Check Ollama logs
  - Verify GGUF compatibility
  - Adjust Modelfile parameters
- [ ] Verify model appears in Ollama list
  ```bash
  ollama list
  ```

---

## Phase 5: Testing & Validation

### 5.1 Basic Functionality Tests
- [ ] Create test script: `tests/test_basic.py`
- [ ] Test text-only prompts
  ```bash
  ollama run smolvlm2 "Hello, who are you?"
  ```
- [ ] Test image analysis
  ```bash
  ollama run smolvlm2 "Describe this image" --image test.jpg
  ```
- [ ] Test various image types (JPG, PNG, WebP)

### 5.2 Performance Testing
- [ ] Benchmark inference speed
  - Tokens per second
  - Time to first token
  - Image encoding time
- [ ] Test with different image sizes
- [ ] Test context length limits
- [ ] Monitor resource usage (RAM, VRAM)

### 5.3 Quality Assessment
- [ ] Test with diverse image types
  - Photos, diagrams, screenshots, charts
- [ ] Compare outputs with HuggingFace API version
- [ ] Document any quality differences
- [ ] Test edge cases (blank images, corrupted images, etc.)

---

## Phase 6: Documentation & Polish

### 6.1 User Documentation
- [ ] Create `docs/USAGE.md`
  - Installation instructions
  - Basic usage examples
  - Image input methods
  - Common prompts and use cases
- [ ] Create example scripts
  - `examples/analyze_image.py`
  - `examples/batch_process.py`
  - `examples/interactive_chat.py`

### 6.2 Technical Documentation
- [ ] Document conversion process in detail
- [ ] Create troubleshooting guide
- [ ] Document hardware requirements
  - Minimum RAM
  - Recommended GPU
  - Disk space
- [ ] Add performance benchmarks

### 6.3 Project Cleanup
- [ ] Update main README.md with SmolVLM2 status
- [ ] Add LICENSE file if needed
- [ ] Create .gitignore for model files
  ```
  models/
  *.gguf
  *.bin
  __pycache__/
  ```

---

## Phase 7: Advanced Features (Optional)

### 7.1 Optimizations
- [ ] Create optimized Modelfiles for different use cases
  - High quality (Q8 quantization)
  - Balanced (Q5 quantization)
  - Fast inference (Q4 quantization)
- [ ] Add custom system prompts for specific tasks
  - OCR and text extraction
  - Image captioning
  - Visual question answering

### 7.2 Integration Examples
- [ ] Create REST API wrapper
- [ ] Create CLI tool for batch processing
- [ ] Add support for video frame analysis

---

## Current Status: Planning Phase

**Next Steps:**
1. Research SmolVLM2 architecture
2. Verify Ollama vision support
3. Set up development environment

## Notes & Blockers

- **Critical**: Ollama must support vision models (check version compatibility)
- **Important**: SmolVLM2 architecture must be compatible with GGUF conversion
- **Risk**: Vision encoder might need separate handling
- **Alternative**: If direct conversion fails, may need to wait for official Ollama support or contribute conversion support to llama.cpp

## Resources

- [SmolVLM2 HuggingFace](https://huggingface.co/HuggingFaceTB/SmolVLM2-Instruct)
- [Ollama Vision Models Guide](https://ollama.com/blog/vision-models)
- [llama.cpp Conversion Guide](https://github.com/ggerganov/llama.cpp/blob/master/docs/GGUF.md)
- [LLaVA Ollama Example](https://ollama.com/library/llava)
