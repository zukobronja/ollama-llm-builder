#!/usr/bin/env python3
"""
Convert HuggingFace models to GGUF format

Educational script demonstrating the GGUF conversion process.
"""

import sys
import logging
import argparse
import subprocess
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.utils.config_manager import ConfigManager


logger = logging.getLogger(__name__)


class GGUFConverter:
    """Convert HuggingFace models to GGUF format."""

    def __init__(self, config: ConfigManager):
        """Initialize converter."""
        self.config = config
        self.paths = config.get_paths()

    def find_llama_cpp(self) -> Path:
        """Find llama.cpp installation."""
        # Check configured path
        llama_cpp_config = self.config.build_config.get('conversion', {}).get('llama_cpp', {})
        llama_cpp_path = Path(llama_cpp_config.get('local_path', './llama.cpp'))

        if llama_cpp_path.exists():
            logger.info(f"Found llama.cpp at: {llama_cpp_path}")
            return llama_cpp_path

        # Check common locations
        common_paths = [
            Path.home() / 'llama.cpp',
            Path('/opt/llama.cpp'),
            Path('/usr/local/llama.cpp'),
        ]

        for path in common_paths:
            if path.exists():
                logger.info(f"Found llama.cpp at: {path}")
                return path

        raise FileNotFoundError(
            "llama.cpp not found. Please install it first:\n"
            "  git clone https://github.com/ggerganov/llama.cpp\n"
            "  cd llama.cpp && make"
        )

    def convert_to_fp16(
        self,
        model_id: str,
        source_dir: Path,
        output_file: Path,
        llama_cpp_path: Path
    ) -> Path:
        """
        Convert HuggingFace model to GGUF FP16 format.

        Args:
            model_id: HuggingFace model ID
            source_dir: Path to downloaded model
            output_file: Output GGUF file path
            llama_cpp_path: Path to llama.cpp installation

        Returns:
            Path to created GGUF file
        """
        logger.info("="*70)
        logger.info("STEP 1: Converting to GGUF FP16 format")
        logger.info("="*70)

        # Find conversion script
        convert_script = llama_cpp_path / 'convert_hf_to_gguf.py'

        if not convert_script.exists():
            raise FileNotFoundError(
                f"Conversion script not found: {convert_script}\n"
                "Make sure you have the latest llama.cpp version."
            )

        # Prepare output directory
        output_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Model: {model_id}")
        logger.info(f"Source: {source_dir}")
        logger.info(f"Output: {output_file}")
        logger.info("")
        logger.info("This may take several minutes...")
        logger.info("")

        # Build conversion command
        cmd = [
            sys.executable,  # Use current Python
            str(convert_script),
            str(source_dir),
            '--outtype', 'f16',
            '--outfile', str(output_file),
        ]

        try:
            logger.info(f"Running: {' '.join(cmd)}")
            logger.info("")

            # Run conversion
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,  # Show output in real-time
                text=True
            )

            # Verify output file
            if not output_file.exists():
                raise FileNotFoundError(f"Conversion failed: {output_file} not created")

            size_gb = output_file.stat().st_size / (1024**3)
            logger.info("")
            logger.info("="*70)
            logger.info("✓ FP16 conversion complete!")
            logger.info(f"Output file: {output_file}")
            logger.info(f"Size: {size_gb:.2f} GB")
            logger.info("="*70)
            logger.info("")

            return output_file

        except subprocess.CalledProcessError as e:
            logger.error("Conversion failed!")
            logger.error(f"Error: {e}")
            logger.error("")
            logger.error("Possible issues:")
            logger.error("  1. Model architecture not supported by llama.cpp")
            logger.error("  2. Corrupted model files")
            logger.error("  3. Insufficient memory")
            logger.error("  4. Outdated llama.cpp version")
            raise

    def quantize(
        self,
        input_file: Path,
        output_file: Path,
        quant_type: str,
        llama_cpp_path: Path
    ) -> Path:
        """
        Quantize GGUF model.

        Args:
            input_file: Input GGUF file (usually FP16)
            output_file: Output quantized file
            quant_type: Quantization type (Q4_K_M, Q5_K_M, Q8_0, etc.)
            llama_cpp_path: Path to llama.cpp installation

        Returns:
            Path to quantized file
        """
        logger.info("="*70)
        logger.info(f"STEP 2: Quantizing to {quant_type}")
        logger.info("="*70)

        # Find quantization tool (check multiple locations)
        possible_paths = [
            llama_cpp_path / 'llama-quantize',
            llama_cpp_path / 'build' / 'bin' / 'llama-quantize',
            llama_cpp_path / 'build' / 'llama-quantize',
        ]

        quantize_tool = None
        for path in possible_paths:
            if path.exists():
                quantize_tool = path
                break

        if not quantize_tool:
            raise FileNotFoundError(
                f"Quantization tool not found. Tried:\n" +
                "\n".join(f"  - {p}" for p in possible_paths) +
                "\n\nMake sure llama.cpp is built (run 'make' in llama.cpp directory)"
            )

        # Prepare output directory
        output_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Input: {input_file}")
        logger.info(f"Output: {output_file}")
        logger.info(f"Type: {quant_type}")
        logger.info("")

        # Get quantization info
        try:
            quant_info = self.config.get_quantization_info(quant_type)
            logger.info(f"Description: {quant_info.get('description')}")
            logger.info(f"Quality: {quant_info.get('quality')}")
            logger.info(f"Speed: {quant_info.get('speed')}")
            logger.info("")
        except:
            pass

        logger.info("Quantizing... this may take a few minutes...")
        logger.info("")

        # Build quantization command
        cmd = [
            str(quantize_tool),
            str(input_file),
            str(output_file),
            quant_type
        ]

        try:
            logger.info(f"Running: {' '.join(cmd)}")
            logger.info("")

            # Run quantization
            subprocess.run(
                cmd,
                check=True,
                capture_output=False,  # Show output in real-time
                text=True
            )

            # Verify and report
            if not output_file.exists():
                raise FileNotFoundError(f"Quantization failed: {output_file} not created")

            input_size = input_file.stat().st_size / (1024**3)
            output_size = output_file.stat().st_size / (1024**3)
            reduction = ((input_size - output_size) / input_size) * 100

            logger.info("")
            logger.info("="*70)
            logger.info(f"✓ Quantization to {quant_type} complete!")
            logger.info(f"Output file: {output_file}")
            logger.info(f"Original size: {input_size:.2f} GB")
            logger.info(f"Quantized size: {output_size:.2f} GB")
            logger.info(f"Size reduction: {reduction:.1f}%")
            logger.info("="*70)
            logger.info("")

            return output_file

        except subprocess.CalledProcessError as e:
            logger.error("Quantization failed!")
            logger.error(f"Error: {e}")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Convert HuggingFace models to GGUF format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert to FP16 only
  python scripts/conversion/convert_to_gguf.py \\
    --model-id HuggingFaceTB/SmolVLM2-2.2B-Instruct \\
    --source models/source/SmolVLM2-2.2B-Instruct

  # Convert and quantize to Q4_K_M
  python scripts/conversion/convert_to_gguf.py \\
    --model-id HuggingFaceTB/SmolVLM2-2.2B-Instruct \\
    --source models/source/SmolVLM2-2.2B-Instruct \\
    --quantize Q4_K_M

  # Create multiple quantizations
  python scripts/conversion/convert_to_gguf.py \\
    --model-id HuggingFaceTB/SmolVLM2-2.2B-Instruct \\
    --source models/source/SmolVLM2-2.2B-Instruct \\
    --quantize Q4_K_M,Q5_K_M,Q8_0
        """
    )
    parser.add_argument(
        '--model-id',
        required=True,
        help='HuggingFace model ID'
    )
    parser.add_argument(
        '--source',
        required=True,
        type=Path,
        help='Path to source model directory'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory for GGUF files (default: models/gguf)'
    )
    parser.add_argument(
        '--quantize',
        help='Quantization types (comma-separated, e.g., Q4_K_M,Q5_K_M,Q8_0)'
    )
    parser.add_argument(
        '--llama-cpp',
        type=Path,
        help='Path to llama.cpp installation (auto-detected if not specified)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    try:
        # Initialize
        config = ConfigManager()
        converter = GGUFConverter(config)

        # Find llama.cpp
        if args.llama_cpp:
            llama_cpp_path = args.llama_cpp
        else:
            llama_cpp_path = converter.find_llama_cpp()

        # Determine output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = Path(config.get_paths().get('gguf_dir', './models/gguf'))

        # Extract model name from ID
        model_name = args.model_id.split('/')[-1]

        # Step 1: Convert to FP16
        fp16_file = output_dir / f"{model_name}-F16.gguf"

        if fp16_file.exists():
            logger.info(f"FP16 file already exists: {fp16_file}")
            logger.info("Skipping conversion (delete file to re-convert)")
        else:
            converter.convert_to_fp16(
                args.model_id,
                args.source,
                fp16_file,
                llama_cpp_path
            )

        # Step 2: Quantize if requested
        if args.quantize:
            quant_types = [q.strip() for q in args.quantize.split(',')]

            for quant_type in quant_types:
                quant_file = output_dir / f"{model_name}-{quant_type}.gguf"

                if quant_file.exists():
                    logger.info(f"{quant_type} file already exists: {quant_file}")
                    logger.info("Skipping quantization")
                    continue

                converter.quantize(
                    fp16_file,
                    quant_file,
                    quant_type,
                    llama_cpp_path
                )

        # Summary
        logger.info("")
        logger.info("="*70)
        logger.info("CONVERSION COMPLETE!")
        logger.info("="*70)
        logger.info("")
        logger.info("Created files:")

        for gguf_file in sorted(output_dir.glob(f"{model_name}*.gguf")):
            size_gb = gguf_file.stat().st_size / (1024**3)
            logger.info(f"  {gguf_file.name} ({size_gb:.2f} GB)")

        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Test with llama.cpp:")
        logger.info(f"     llama-cli --model {output_dir}/{model_name}-Q4_K_M.gguf")
        logger.info("")
        logger.info("  2. Create Ollama Modelfile:")
        logger.info("     python scripts/generate_modelfile.py ...")
        logger.info("")

        return 0

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
