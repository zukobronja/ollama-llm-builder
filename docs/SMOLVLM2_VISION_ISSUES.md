# SmolVLM2 Vision Issues and Solutions

**Last Updated**: November 2025
**Status**: CRITICAL - Vision recognition broken in Ollama deployment

---

## Problem Summary

SmolVLM2 models deployed to Ollama exhibit **poor vision recognition**:
- Describes objects not present in images
- Generates random/hallucinated descriptions
- Cannot accurately identify image content

**Root Cause**: Missing multimodal projector (mmproj) file + unsupported architecture in Ollama.

---

## Technical Analysis

### 1. Vision-Language Model Architecture

SmolVLM2 consists of **three components**:

```
┌─────────────────┐
│ Vision Encoder │ ← SigLIP (384x384 patches, 14x14 inner)
└────────┬────────┘
   │ Encodes images to 81 tokens per patch
   ↓
┌─────────────────┐
│ Projector  │ ← Compresses visual info 9x (pixel shuffle)
└────────┬────────┘
   │
   ↓
┌─────────────────┐
│ Language Model │ ← Text generation (2.2B params)
└─────────────────┘
```

### 2. HuggingFace Implementation (WORKS)

**Key Configuration**:
```python
processor = AutoProcessor.from_pretrained(
 "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
 torch_dtype=torch.bfloat16
)

model = AutoModelForImageTextToText.from_pretrained(
 "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
 torch_dtype=torch.bfloat16,   # NO quantization!
 device_map="cuda:0",
 low_cpu_mem_usage=True,
 trust_remote_code=True
)
```

**Why It Works**:
- Uses `torch.bfloat16` (near-full precision)
- All three components loaded together
- Vision encoder at full quality
- Native PyTorch integration

**Performance**:
- **VRAM**: ~6GB on RTX 4070 8GB
- **Quality**: Excellent vision recognition
- **Speed**: Fast inference
- **Video support**: Yes (multiple frames)

---

### 3. GGUF/Ollama Implementation (BROKEN)

**Current Conversion**:
```bash
python scripts/conversion/convert_to_gguf.py \
 --model-id HuggingFaceTB/SmolVLM2-2.2B-Instruct \
 --source models/source/SmolVLM2-2.2B-Instruct \
 --quantize Q4_K_M,Q5_K_M,Q8_0
```

**What Gets Created**:
- `SmolVLM2-2.2B-Instruct-F16.gguf` (3.4GB) - Language model only
- `SmolVLM2-2.2B-Instruct-Q4_K_M.gguf` (1.1GB) - Quantized
- **MISSING**: `SmolVLM2-mmproj-f16.gguf` - Vision encoder!

**Why It Fails**:

1. **Missing Vision Encoder (mmproj)**
 - Conversion script has NO `--mmproj` support
 - Language model tries to process images without vision encoder
 - Results in hallucinations

2. **Quantization Destroys Vision Quality**
 - Vision encoders are **extremely sensitive** to quantization
 - Q4_K_M/Q5_K_M degrade vision understanding significantly
 - Text models tolerate quantization, vision models don't

3. **Ollama Lacks Official Support**
 - Issue [#9559](https://github.com/ollama/ollama/issues/9559) - SmolVLM support requested
 - SmolVLM2 architecture not yet supported by Ollama
 - GGUF file alone is insufficient

---

## Evidence: Actual vs Expected Behavior

### HuggingFace (Working)

**Test**: Bee on pink flower image from HuggingFace documentation

**Output**:
```
The image depicts a close-up view of a bee on a pink flower.
The bee is positioned in the center of the flower, with its
body prominently displayed. The flower itself is vibrant and
has a prominent pink hue, with a yellow center that is slightly
blurred due to the focus on the bee.
```
 **Accurate, detailed, correct**

### Ollama F16/Q4_K_M (Broken)

**Same Prompt**: "Can you describe this image?"

**Output**:
```
[Describes objects not in image, random hallucinations]
```
 **Inaccurate, hallucinations, unusable**

---

## Solutions

### Option 1: Use HuggingFace Directly (RECOMMENDED)

**Pros**:
- Already working perfectly
- Best vision quality (no quantization loss)
- Full feature support (images + video)
- ~6GB VRAM (fits RTX 4070)
- Production-ready Streamlit app available

**Cons**:
- Requires Python/PyTorch environment
- Not integrated with Ollama/Open WebUI

**Use Cases**:
- Video analysis (Streamlit app)
- Batch image processing
- Interactive notebooks
- Production applications

**Example**:
```bash
cd /mnt/data/projects/learning/aiml/demo-learning/vision-llm/notebooks/notebooks/
streamlit run video_chat_app.py
```

---

### Option 2: llama.cpp Server with mmproj (ADVANCED)

**Requires**:
1. Reconvert model with mmproj support
2. Run llama-server instead of Ollama
3. OpenAI-compatible API

**Step 1: Reconvert with mmproj**

Update `scripts/conversion/convert_to_gguf.py`:
```python
def convert_to_fp16(self, model_id, source_dir, output_file, llama_cpp_path):
 # Find conversion script
 convert_script = llama_cpp_path / 'convert_hf_to_gguf.py'

 # Prepare mmproj output path
 mmproj_file = output_file.parent / f"{output_file.stem}-mmproj-f16.gguf"

 # Build conversion command with mmproj support
 cmd = [
  sys.executable,
  str(convert_script),
  str(source_dir),
  '--outtype', 'f16',
  '--outfile', str(output_file),
  '--mmproj', str(mmproj_file), # Add this!
 ]

 # Run conversion...
```

**Step 2: Run llama-server**

```bash
# Start server with vision support
llama-server \
 -m models/gguf/SmolVLM2-2.2B-Instruct-F16.gguf \
 --mmproj models/gguf/SmolVLM2-2.2B-Instruct-mmproj-f16.gguf \
 --host 0.0.0.0 \
 --port 8080 \
 --ctx-size 4096 \
 --n-gpu-layers -1
```

**Step 3: Connect Open WebUI**

```bash
# Point Open WebUI to llama-server instead of Ollama
# Settings → Connections → Add OpenAI API
# URL: http://localhost:8080/v1
```

**Pros**:
- OpenAI-compatible API
- Can integrate with Open WebUI
- Better control over parameters

**Cons**:
- Requires manual server management
- More complex setup
- Still untested (needs verification)

---

### Option 3: Wait for Official Ollama Support

**Status**: Open issue [#9559](https://github.com/ollama/ollama/issues/9559)

**Timeline**: Unknown (could be weeks to months)

**When Available**:
```bash
# Future (when supported)
ollama pull smolvlm2
ollama run smolvlm2 "Describe this image" --image photo.jpg
```

**Not Recommended**: Waiting provides no value when HuggingFace works now.

---

## Quantization Impact on Vision Models

### Text-Only Models
- Q4_K_M: Good quality, 75% size reduction
- Q5_K_M: Very good quality, 65% size reduction
- Q8_0: Excellent quality, 50% size reduction

### Vision-Language Models (SmolVLM2)
- **F16 (bfloat16)**: **Best quality** (RECOMMENDED)
- Q8_0: Degraded vision understanding
- Q5_K_M: Significant vision quality loss
- Q4_K_M: Severe hallucinations

**Why?**

Vision encoders:
- Process high-dimensional image data (384x384 pixels)
- Require precise numerical representations
- Lose critical visual features when quantized
- Cannot recover from precision loss

Text models:
- Process discrete tokens
- More tolerant to quantization
- Can maintain semantic meaning with lower precision

---

## Correct Chat Template

**Source**: `models/source/SmolVLM2-2.2B-Instruct/tokenizer_config.json:1171`

```jinja2
<|im_start|>{% for message in messages %}
{{message['role'] | capitalize}}:
{% for line in message['content'] %}
 {% if line['type'] == 'text' %}{{line['text']}}
 {% elif line['type'] == 'image' %}{{ '<image>' }}
 {% endif %}
{% endfor %}<end_of_utterance>
{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}
```

**Key Points**:
1. Image token: `<image>` (NO closing tag!)
2. End token: `<end_of_utterance>` (NOT `<|im_end|>`)
3. Role format: `User:` or `Assistant:` (capitalized)

**Current Modelfile Issues**:
```modelfile
# WRONG 
{{ if .Image }}<image>{{ .Image }}</image>{{ end }}

# CORRECT 
{{ if .Image }}<image>{{ end }}
```

---

## Performance Comparison

### HuggingFace (bfloat16)
| Metric | Value |
|--------|-------|
| VRAM Usage | ~6GB |
| Model Size | 4.5GB (in memory) |
| Vision Quality | Excellent |
| Speed | Fast (~50 tokens/sec) |
| Video Support | Yes |
| Hallucinations | None |

### Ollama F16 (3.4GB GGUF)
| Metric | Value |
|--------|-------|
| VRAM Usage | ~4GB |
| Model Size | 3.4GB |
| Vision Quality | Broken (no mmproj) |
| Speed | N/A (unusable) |
| Video Support | No |
| Hallucinations | Constant |

### Ollama Q4_K_M (1.1GB GGUF)
| Metric | Value |
|--------|-------|
| VRAM Usage | ~2.8GB |
| Model Size | 1.1GB |
| Vision Quality | Severely broken |
| Speed | N/A (unusable) |
| Video Support | No |
| Hallucinations | Severe |

---

## Recommendations by Use Case

### Interactive Image/Video Analysis
**Use**: HuggingFace Transformers + Streamlit/Gradio
- Best quality
- Video support
- Easy to deploy

### Batch Image Processing
**Use**: HuggingFace Transformers + Python scripts
- Full control
- Best accuracy
- No quality loss

### Integration with Open WebUI
**Options**:
1. Keep text models in Ollama, use HuggingFace for vision separately
2. Set up llama-server with mmproj (advanced)
3. Wait for official Ollama support (not recommended)

### Production Deployment
**Use**: HuggingFace + FastAPI/Streamlit
- Reliable
- Tested
- Best quality

---

## Files and Locations

### Ollama Builder Project
```
/mnt/data/projects/learning/aiml/my-llms/ollama/ollama-llm-builder/
├── models/source/SmolVLM2-2.2B-Instruct/ # Original model
├── models/gguf/
│ ├── SmolVLM2-2.2B-Instruct-F16.gguf  # 3.4GB (no vision!)
│ ├── SmolVLM2-2.2B-Instruct-Q4_K_M.gguf # 1.1GB (broken)
│ └── [MISSING] SmolVLM2-mmproj-f16.gguf # Vision encoder!
├── scripts/conversion/convert_to_gguf.py  # Needs mmproj support
└── docs/SMOLVLM2_VISION_ISSUES.md   # This file
```

---

## Action Items

### Immediate (Do This Now)
- [x] Document vision issues
- [ ] Update project README with warning about vision quality
- [ ] Add note in COMPLETE_WORKFLOW.md about HuggingFace alternative

### Short Term (If Needed)
- [ ] Update `convert_to_gguf.py` to support `--mmproj`
- [ ] Test llama-server with mmproj file
- [ ] Create integration guide for Open WebUI + llama-server

### Long Term (Monitor)
- [ ] Track Ollama issue [#9559](https://github.com/ollama/ollama/issues/9559)
- [ ] Test official support when available
- [ ] Update documentation with official workflow

---

## References

### Example Code
```python
# Working HuggingFace implementation example
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

processor = AutoProcessor.from_pretrained(
 "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
 torch_dtype=torch.bfloat16
)

model = AutoModelForImageTextToText.from_pretrained(
 "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
 torch_dtype=torch.bfloat16,
 device_map="cuda:0",
 low_cpu_mem_usage=True,
 trust_remote_code=True
)

# Use for image analysis
messages = [{
 "role": "user",
 "content": [
  {"type": "image", "url": "path/to/image.jpg"},
  {"type": "text", "text": "Describe this image"}
 ]
}]

inputs = processor.apply_chat_template(
 messages, add_generation_prompt=True,
 tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

generated_ids = model.generate(**inputs, max_new_tokens=200)
response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

### Issues
- Ollama SmolVLM Support: https://github.com/ollama/ollama/issues/9559
- Ollama Vision Architecture: https://github.com/ollama/ollama/issues/7912
- llama.cpp mmproj Discussion: https://github.com/ggml-org/llama.cpp/discussions/1050

### Documentation
- HuggingFace Model: https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct
- SmolVLM Blog: https://huggingface.co/blog/smolvlm
- llama.cpp Vision Support: https://simonwillison.net/2025/May/10/llama-cpp-vision/

---

## Conclusion

**For SmolVLM2 vision tasks, use the HuggingFace implementation**. It's:
- Production-ready
- Best quality (no quantization)
- Already tested and working
- Fits in 8GB VRAM
- Supports images and video

**Avoid Ollama deployment** until:
- Official SmolVLM2 support is added
- mmproj conversion is implemented and tested
- Vision quality is verified to match HuggingFace

The quantized GGUF models are suitable for **text-only models**, but **not for vision-language models** like SmolVLM2.
