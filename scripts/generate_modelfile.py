#!/usr/bin/env python3
"""
Generate Ollama Modelfile for GGUF models

Creates optimized Modelfiles based on hardware profiles.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utils.config_manager import ConfigManager


def generate_modelfile(
    model_id: str,
    gguf_file: Path,
    output_file: Path,
    config: ConfigManager,
    profile_name: str = None
) -> Path:
    """
    Generate Ollama Modelfile.

    Args:
        model_id: HuggingFace model ID
        gguf_file: Path to GGUF file
        output_file: Where to save Modelfile
        config: ConfigManager instance
        profile_name: Hardware profile name

    Returns:
        Path to created Modelfile
    """
    # Get hardware profile
    if profile_name:
        profile = config.get_profile(profile_name)
    else:
        profile = config.get_active_profile()
        profile_name = config.build_config.get('active_profile')

    # Get Ollama parameters
    ollama_params = config.get_ollama_params(profile_name)

    # Extract model name
    model_name = model_id.split('/')[-1]

    # Generate Modelfile content
    modelfile_content = f"""# Modelfile for {model_name}
# Generated for hardware profile: {profile_name}
# Profile: {profile.get('description')}

FROM {gguf_file.absolute()}

# Temperature controls randomness (0.0 = deterministic, 1.0 = creative)
PARAMETER temperature {ollama_params.get('temperature', 0.7)}

# Top-p sampling (0.9 = use top 90% probability mass)
PARAMETER top_p {ollama_params.get('top_p', 0.9)}

# Context window size
PARAMETER num_ctx {ollama_params.get('num_ctx', 4096)}

# Number of GPU layers (-1 = all layers on GPU)
PARAMETER num_gpu {ollama_params.get('num_gpu_layers', -1)}

# Stop tokens
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<end_of_utterance>"

# System prompt
SYSTEM \"\"\"You are SmolVLM2, a vision-language AI assistant. You can analyze images and answer questions about them. Be helpful, accurate, and concise.\"\"\"

# Template for vision-language interaction
TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}<|im_start|>user
{{{{ if .Prompt }}}}{{{{ .Prompt }}}}{{{{ end }}}}{{{{ if .Image }}}}<image>{{{{ .Image }}}}</image>{{{{ end }}}}<|im_end|>
<|im_start|>assistant
\"\"\"
"""

    # Write Modelfile
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(modelfile_content)

    return output_file


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate Ollama Modelfile for GGUF models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate Modelfile for Q4_K_M
  python scripts/generate_modelfile.py \\
    --model-id HuggingFaceTB/SmolVLM2-2.2B-Instruct \\
    --gguf models/gguf/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf

  # Generate for specific hardware profile
  python scripts/generate_modelfile.py \\
    --model-id HuggingFaceTB/SmolVLM2-2.2B-Instruct \\
    --gguf models/gguf/SmolVLM2-2.2B-Instruct-Q8_0.gguf \\
    --profile workstation_rtx4090
        """
    )
    parser.add_argument(
        '--model-id',
        required=True,
        help='HuggingFace model ID'
    )
    parser.add_argument(
        '--gguf',
        required=True,
        type=Path,
        help='Path to GGUF file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output Modelfile path (default: modelfiles/<name>.modelfile)'
    )
    parser.add_argument(
        '--profile',
        help='Hardware profile to use (default: active profile)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Verify GGUF file exists
    if not args.gguf.exists():
        print(f"Error: GGUF file not found: {args.gguf}")
        return 1

    # Initialize config
    config = ConfigManager()

    # Determine output path
    if args.output:
        output_file = args.output
    else:
        # Extract model and quantization from filename
        # e.g., SmolVLM2-2.2B-Instruct-Q4_K_M.gguf -> SmolVLM2-Q4_K_M
        filename = args.gguf.stem
        model_short = filename.split('-')[0]
        quant = filename.split('-')[-1] if '-' in filename else 'F16'
        output_file = Path('modelfiles') / f"{model_short}-{quant}.modelfile"

    print(f"Generating Modelfile...")
    print(f"  Model: {args.model_id}")
    print(f"  GGUF: {args.gguf}")
    print(f"  Output: {output_file}")

    # Generate Modelfile
    modelfile = generate_modelfile(
        args.model_id,
        args.gguf,
        output_file,
        config,
        args.profile
    )

    # Extract quantization from filename for tag suggestion
    quant = args.gguf.stem.split('-')[-1].lower()

    print(f"\n✓ Modelfile created: {modelfile}")
    print(f"\nNext steps:")
    print(f"  1. Create Ollama model with tag (recommended):")
    print(f"     ollama create smolvlm2:{quant} -f {modelfile}")
    print(f"  2. Or create without tag (becomes 'latest'):")
    print(f"     ollama create smolvlm2 -f {modelfile}")
    print(f"  3. Test the model:")
    print(f"     ollama run smolvlm2:{quant} 'Hello, who are you?'")
    print(f"\nPro Tip: Use tags to keep multiple quantizations:")
    print(f"  ollama create smolvlm2:q4_k_m -f modelfiles/SmolVLM2-Q4_K_M.modelfile")
    print(f"  ollama create smolvlm2:q5_k_m -f modelfiles/SmolVLM2-Q5_K_M.modelfile")
    print(f"  ollama create smolvlm2:q8_0 -f modelfiles/SmolVLM2-Q8_0.modelfile")
    print()

    if args.verbose:
        print("Modelfile content:")
        print("=" * 70)
        print(modelfile.read_text())
        print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
