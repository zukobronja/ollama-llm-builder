# Ollama Model Tags Guide

## Why Use Tags?

Tags allow you to keep **multiple versions** of the same model with different quantizations or configurations.

## Tag Format

```
model_name:tag_name
```

Examples:
- `smolvlm2:q4_k_m` - Q4_K_M quantization
- `smolvlm2:q5_k_m` - Q5_K_M quantization
- `smolvlm2:q8_0` - Q8_0 quantization
- `smolvlm2:f16` - Full precision
- `smolvlm2:latest` - Default version

## Creating Models with Tags

### Individual Tags

```fish
# Create Q4_K_M version
ollama create smolvlm2:q4_k_m -f modelfiles/SmolVLM2-Q4_K_M.modelfile

# Create Q5_K_M version
ollama create smolvlm2:q5_k_m -f modelfiles/SmolVLM2-Q5_K_M.modelfile

# Create Q8_0 version
ollama create smolvlm2:q8_0 -f modelfiles/SmolVLM2-Q8_0.modelfile

# Create F16 version
ollama create smolvlm2:f16 -f modelfiles/SmolVLM2-F16.modelfile
```

### Set Default (Latest)

```fish
# Make Q4_K_M the default
ollama create smolvlm2:latest -f modelfiles/SmolVLM2-Q4_K_M.modelfile

# Or just use the name without a tag (becomes 'latest')
ollama create smolvlm2 -f modelfiles/SmolVLM2-Q4_K_M.modelfile
```

## Using Tagged Models

### Run Specific Version

```fish
# Use Q4_K_M (fast, low VRAM)
ollama run smolvlm2:q4_k_m "Your prompt here"

# Use Q8_0 (best quality)
ollama run smolvlm2:q8_0 "Your prompt here"
```

### Run Default Version

```fish
# Uses the 'latest' tag
ollama run smolvlm2 "Your prompt here"
```

## Managing Tags

### List All Versions

```fish
ollama list
```

Output:
```
NAME    ID    SIZE  MODIFIED
smolvlm2:q4_k_m  abc123   1.1 GB 5 minutes ago
smolvlm2:q5_k_m  def456   1.3 GB 4 minutes ago
smolvlm2:q8_0  ghi789   1.8 GB 3 minutes ago
smolvlm2:latest  abc123   1.1 GB 2 minutes ago
```

### Remove Specific Tag

```fish
ollama rm smolvlm2:q8_0
```

### Remove All Versions

```fish
ollama rm smolvlm2:q4_k_m
ollama rm smolvlm2:q5_k_m
ollama rm smolvlm2:q8_0
ollama rm smolvlm2:latest
```

## Recommended Tag Naming

### Quantization Tags

Use lowercase, match the quantization type:

- `f16` - Full precision FP16
- `q8_0` - Q8_0 quantization
- `q6_k` - Q6_K quantization
- `q5_k_m` - Q5_K_M quantization
- `q4_k_m` - Q4_K_M quantization
- `q4_0` - Q4_0 quantization

### Custom Tags

You can also use custom tags:

- `smolvlm2:fast` - For Q4_K_M (fastest)
- `smolvlm2:quality` - For Q8_0 (best quality)
- `smolvlm2:balanced` - For Q5_K_M (balanced)
- `smolvlm2:reference` - For F16 (reference)

## Comparison Workflow

Having multiple tagged versions lets you easily compare:

```fish
# Compare quality
echo "Test prompt: Describe artificial intelligence" > prompt.txt

ollama run smolvlm2:q4_k_m < prompt.txt > output_q4.txt
ollama run smolvlm2:q5_k_m < prompt.txt > output_q5.txt
ollama run smolvlm2:q8_0 < prompt.txt > output_q8.txt

# Compare outputs
diff output_q4.txt output_q5.txt
diff output_q5.txt output_q8.txt
```

## Automated Deployment Script

Create all versions at once:

```fish
#!/usr/bin/env fish

# Deploy all SmolVLM2 quantizations with tags

set QUANTS q4_k_m q5_k_m q8_0 f16

for quant in $QUANTS
 set quant_upper (echo $quant | tr '[:lower:]' '[:upper:]')
 echo "Creating smolvlm2:$quant from SmolVLM2-$quant_upper.modelfile"

 ollama create smolvlm2:$quant -f modelfiles/SmolVLM2-$quant_upper.modelfile
end

# Set Q4_K_M as default
echo "Setting smolvlm2:latest to Q4_K_M"
ollama create smolvlm2:latest -f modelfiles/SmolVLM2-Q4_K_M.modelfile

echo "All models deployed!"
ollama list | grep smolvlm2
```

Save as `deploy_all.fish`, then:

```fish
chmod +x deploy_all.fish
./deploy_all.fish
```

## Best Practices

1. **Use descriptive tags** - Make it clear what each version is
2. **Set a default** - Tag your most-used version as `latest`
3. **Document your tags** - Keep a list of what each tag represents
4. **Clean up unused versions** - Remove old tags to save disk space

## Tag Conventions

### For Hardware

```
smolvlm2:laptop # Optimized for laptop (Q4_K_M)
smolvlm2:desktop # Optimized for desktop (Q5_K_M)
smolvlm2:server # Optimized for server (Q8_0)
```

### For Use Case

```
smolvlm2:draft # Quick drafts (Q4_K_M)
smolvlm2:final # Final output (Q8_0)
smolvlm2:test  # Testing version
```

### For Experiments

```
smolvlm2:exp-1 # Experiment 1
smolvlm2:exp-2 # Experiment 2
smolvlm2:baseline # Baseline for comparison
```

## Examples

### Development Workflow

```fish
# Development/testing (fast)
ollama run smolvlm2:q4_k_m "Quick test prompt"

# Production (quality)
ollama run smolvlm2:q8_0 "Production prompt"

# Benchmark/reference
ollama run smolvlm2:f16 "Reference output"
```

### A/B Testing

```fish
# Test same prompt with different quantizations
set PROMPT "Explain quantum computing in simple terms"

echo "Q4_K_M output:"
ollama run smolvlm2:q4_k_m "$PROMPT"

echo "\nQ5_K_M output:"
ollama run smolvlm2:q5_k_m "$PROMPT"

echo "\nQ8_0 output:"
ollama run smolvlm2:q8_0 "$PROMPT"
```

## Summary

- Use tags to keep multiple versions
- Tag format: `model:tag`
- Set a default with `latest`
- Easy to compare and switch
- Saves time and disk space
