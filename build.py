#!/usr/bin/env python3
"""
Ollama LLM Builder - Main Build Script

Orchestrates the entire process of downloading, converting, and deploying
models to Ollama with configurable hardware profiles.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from utils.config_manager import ConfigManager


def setup_logging(level: str = "INFO", log_file: str = None):
    """Set up logging configuration."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    return logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Build and deploy LLMs to Ollama',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build SmolVLM2 with default profile (laptop)
  python build.py --model HuggingFaceTB/SmolVLM2-2.2B-Instruct

  # Build for a specific hardware profile
  python build.py --model HuggingFaceTB/SmolVLM2-2.2B-Instruct --profile workstation_rtx4090

  # List available profiles and recommended models
  python build.py --list-profiles
  python build.py --list-models

  # Check compatibility
  python build.py --model HuggingFaceTB/SmolVLM2-2.2B-Instruct --check-compatibility

  # Build with custom quantization
  python build.py --model HuggingFaceTB/SmolVLM2-500M-Video-Instruct --quantization Q5_K_M,Q8_0

  # Skip conversion if model exists
  python build.py --model HuggingFaceTB/SmolVLM2-2.2B-Instruct --skip-existing
        """
    )

    parser.add_argument(
        '--model', '-m',
        help='HuggingFace model ID (e.g., HuggingFaceTB/SmolVLM2-2.2B-Instruct)'
    )

    parser.add_argument(
        '--profile', '-p',
        help='Hardware profile to use (default: from config)'
    )

    parser.add_argument(
        '--quantization', '-q',
        help='Quantization levels (comma-separated, e.g., Q4_K_M,Q5_K_M)'
    )

    parser.add_argument(
        '--list-profiles',
        action='store_true',
        help='List available hardware profiles'
    )

    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available models'
    )

    parser.add_argument(
        '--check-compatibility',
        action='store_true',
        help='Check hardware compatibility without building'
    )

    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip download/conversion if files exist'
    )

    parser.add_argument(
        '--no-cleanup',
        action='store_true',
        help='Keep intermediate files after build'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    return parser.parse_args()


def list_profiles(config: ConfigManager):
    """List all available hardware profiles."""
    print("\n" + "=" * 70)
    print("AVAILABLE HARDWARE PROFILES")
    print("=" * 70 + "\n")

    for profile_name in config.list_profiles():
        profile = config.get_profile(profile_name)
        is_active = profile_name == config.build_config.get('active_profile')
        marker = " [ACTIVE]" if is_active else ""

        print(f"{profile_name}{marker}")
        print(f"  {profile.get('description')}")
        print(f"  GPU: {profile['specs'].get('gpu', 'N/A')}")
        print(f"  VRAM: {profile['specs'].get('vram_gb', 0)}GB")
        print(f"  RAM: {profile['specs'].get('ram_gb', 0)}GB")
        print(f"  Recommended: {', '.join(profile.get('recommended_quantization', []))}")
        print()


def list_models(config: ConfigManager):
    """List recommended models."""
    print("\n" + "=" * 70)
    print("RECOMMENDED MODELS")
    print("=" * 70 + "\n")

    models = config.hardware_profiles.get('recommended_models', [])
    if not models:
        print("No recommended models configured.")
        print("Use any HuggingFace model ID directly, for example:")
        print("  python build.py --model HuggingFaceTB/SmolVLM2-2.2B-Instruct")
        return

    for model in models:
        print(f"{model.get('model_id')}")
        print(f"  {model.get('description')}")
        print(f"  Size: ~{model.get('size_gb')}GB (FP16)")
        print(f"  Min VRAM: {model.get('min_vram_gb')}GB (with quantization)")
        print(f"  Min RAM: {model.get('min_ram_gb')}GB")
        print()


def check_compatibility(config: ConfigManager, model_id: str, profile_name: str = None):
    """Check hardware compatibility for a model."""
    print("\n" + "=" * 70)
    print("HARDWARE COMPATIBILITY CHECK")
    print("=" * 70 + "\n")

    if profile_name:
        profile = config.get_profile(profile_name)
    else:
        profile = config.get_active_profile()
        profile_name = config.build_config.get('active_profile')

    # Try to find model in recommended list, otherwise use defaults
    recommended = config.hardware_profiles.get('recommended_models', [])
    model_info = next((m for m in recommended if m['model_id'] == model_id), None)

    if model_info:
        base_size = model_info.get('size_gb', 2.0)
        min_vram = model_info.get('min_vram_gb', 4)
        min_ram = model_info.get('min_ram_gb', 16)
    else:
        # Default estimates for unknown models
        base_size = 2.0
        min_vram = 4
        min_ram = 16
        print("WARNING: Model not in recommended list, using default estimates")

    print(f"Model: {model_id}")
    print(f"Profile: {profile_name} - {profile.get('description')}")
    print()

    # Check compatibility
    specs = profile.get('specs', {})
    available_ram = specs.get('ram_gb', 0)
    available_vram = specs.get('vram_gb', 0)

    compatible = True
    warnings = []

    if min_ram > available_ram:
        compatible = False
        warnings.append(f"Insufficient RAM: {min_ram}GB required, {available_ram}GB available")

    if available_vram == 0:
        warnings.append("No GPU detected - will use CPU-only mode")
    elif min_vram > available_vram:
        warnings.append(f"VRAM constraint: {min_vram}GB minimum, {available_vram}GB available")

    if compatible:
        print("COMPATIBLE")
    else:
        print("NOT COMPATIBLE")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  WARNING: {warning}")

    print("\nRecommended Quantization Levels:")
    for quant in profile.get('recommended_quantization', ['Q4_K_M']):
        quant_info = config.get_quantization_info(quant)
        estimated_vram = config.estimate_vram_usage(base_size, quant)
        print(f"  {quant}: {quant_info.get('description')}")
        print(f"    Estimated VRAM: ~{estimated_vram:.1f}GB")
        print(f"    Quality: {quant_info.get('quality')}")
        print(f"    Speed: {quant_info.get('speed')}")
    print()


def main():
    """Main entry point."""
    args = parse_args()

    # Initialize config manager
    try:
        config = ConfigManager()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you're running this from the project root directory.")
        sys.exit(1)

    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    log_file = config.build_config.get('logging', {}).get('log_file')
    logger = setup_logging(log_level, log_file)

    # Handle list commands
    if args.list_profiles:
        list_profiles(config)
        return 0

    if args.list_models:
        list_models(config)
        return 0

    # Require model for other operations
    if not args.model:
        print("Error: --model is required")
        print("Use --list-models to see recommended models")
        print("\nExample:")
        print("  python build.py --model HuggingFaceTB/SmolVLM2-2.2B-Instruct")
        sys.exit(1)

    # Determine profile
    profile_name = args.profile or config.build_config.get('active_profile')

    # Handle compatibility check
    if args.check_compatibility:
        check_compatibility(config, args.model, profile_name)
        return 0

    # Start build process
    logger.info("=" * 70)
    logger.info("OLLAMA LLM BUILDER")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Profile: {profile_name}")

    # Determine quantization levels
    if args.quantization:
        quant_levels = [q.strip() for q in args.quantization.split(',')]
    else:
        # Use from config or profile recommendations
        build_quant = config.build_config.get('conversion', {}).get('quantization_levels', [])
        if build_quant:
            quant_levels = build_quant
        else:
            profile = config.get_profile(profile_name)
            quant_levels = profile.get('recommended_quantization', ['Q4_K_M'])

    logger.info(f"Quantization levels: {', '.join(quant_levels)}")

    # TODO: Implement actual build steps
    logger.info("\n" + "=" * 70)
    logger.info("BUILD PHASES")
    logger.info("=" * 70)
    logger.info("\nPhase 1: Download model from HuggingFace")
    logger.info("  Status: NOT YET IMPLEMENTED")
    logger.info("  Next: Implement download_model.py script")

    logger.info("\nPhase 2: Convert to GGUF format")
    logger.info("  Status: NOT YET IMPLEMENTED")
    logger.info("  Next: Implement convert_to_gguf.py script")

    logger.info("\nPhase 3: Quantize model")
    logger.info("  Status: NOT YET IMPLEMENTED")
    logger.info("  Next: Implement quantization in conversion script")

    logger.info("\nPhase 4: Generate Modelfile")
    logger.info("  Status: NOT YET IMPLEMENTED")
    logger.info("  Next: Implement generate_modelfile.py script")

    logger.info("\nPhase 5: Create Ollama model")
    logger.info("  Status: NOT YET IMPLEMENTED")
    logger.info("  Next: Implement ollama create command")

    logger.info("\nPhase 6: Test model")
    logger.info("  Status: NOT YET IMPLEMENTED")
    logger.info("  Next: Implement test scripts")

    logger.info("\n" + "=" * 70)
    logger.info("Build configuration created successfully!")
    logger.info("Next steps:")
    logger.info("1. Review config/hardware_profiles.yaml")
    logger.info("2. Review config/build_config.yaml")
    logger.info("3. Implement download and conversion scripts")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
