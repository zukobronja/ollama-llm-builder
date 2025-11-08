#!/usr/bin/env python3
"""
Download model from HuggingFace

Downloads model weights, config, and tokenizer files from HuggingFace Hub.
"""

import sys
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv
import os

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utils.config_manager import ConfigManager


logger = logging.getLogger(__name__)


class ModelDownloader:
    """Download models from HuggingFace Hub."""

    def __init__(self, config: ConfigManager):
        """
        Initialize ModelDownloader.

        Args:
            config: ConfigManager instance
        """
        self.config = config
        self.paths = config.get_paths()
        self.hf_config = config.build_config.get('huggingface', {})

    def download_model(
        self,
        huggingface_id: str,
        force_download: bool = False,
        skip_if_exists: bool = True
    ) -> Path:
        """
        Download a model from HuggingFace.

        Args:
            huggingface_id: HuggingFace model ID (e.g., 'HuggingFaceTB/SmolVLM2-2.2B-Instruct')
            force_download: Force re-download even if cached
            skip_if_exists: Skip download if model already exists

        Returns:
            Path to downloaded model directory
        """

        # Determine download path
        # Extract model name from HuggingFace ID (e.g., "SmolVLM2-2.2B-Instruct" from "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
        model_name = huggingface_id.split('/')[-1]
        source_dir = Path(self.paths.get('source_dir', './models/source'))
        model_dir = source_dir / model_name

        # Create directory
        source_dir.mkdir(parents=True, exist_ok=True)

        # Check if already downloaded
        if skip_if_exists and model_dir.exists() and self._is_valid_download(model_dir):
            logger.info(f"Model already downloaded at {model_dir}")
            logger.info("Use --force-download to re-download")
            return model_dir

        logger.info(f"Downloading model: {huggingface_id}")
        logger.info(f"Destination: {model_dir}")

        try:
            # Get HuggingFace token from environment or config
            # Priority: 1. HF_TOKEN env var, 2. config file
            token = os.getenv('HF_TOKEN') or (
                self.hf_config.get('token') if self.hf_config.get('use_auth_token') else None
            )

            if token:
                logger.info("Using HuggingFace authentication token")
            else:
                logger.info("No HuggingFace token found (using public access)")

            cache_dir = self.hf_config.get('cache_dir', './cache/huggingface')

            # Download using snapshot_download for complete model
            downloaded_path = snapshot_download(
                repo_id=huggingface_id,
                cache_dir=cache_dir,
                local_dir=model_dir,
                local_dir_use_symlinks=False,  # Copy files instead of symlinks
                token=token,
                resume_download=not force_download,
                # Don't download large files we don't need yet
                ignore_patterns=[
                    "*.gguf",  # Skip any pre-converted GGUF files
                    "*.msgpack",
                    "*.h5",
                    "*.ot",
                    "*.md",  # Skip markdown files for now
                ]
            )

            logger.info(f"✓ Model downloaded successfully to {model_dir}")

            # Log what was downloaded
            self._log_download_info(model_dir)

            return Path(downloaded_path)

        except HfHubHTTPError as e:
            logger.error(f"Failed to download model: {e}")
            if "401" in str(e) or "403" in str(e):
                logger.error("Authentication error. Check your HuggingFace token.")
                logger.error("Set 'use_auth_token: true' in config/build_config.yaml")
                logger.error("And add your token: 'token: hf_...'")
            raise
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            raise

    def _is_valid_download(self, model_dir: Path) -> bool:
        """
        Check if a model directory contains a valid download.

        Args:
            model_dir: Path to model directory

        Returns:
            True if valid download exists
        """
        # Check for essential files
        required_files = ['config.json']

        for file in required_files:
            if not (model_dir / file).exists():
                logger.warning(f"Missing required file: {file}")
                return False

        # Check for model weights (either safetensors or pytorch)
        has_safetensors = list(model_dir.glob("*.safetensors"))
        has_pytorch = list(model_dir.glob("*.bin"))

        if not has_safetensors and not has_pytorch:
            logger.warning("No model weight files found")
            return False

        return True

    def _log_download_info(self, model_dir: Path):
        """Log information about downloaded files."""
        logger.info("\nDownloaded files:")

        # Count file types
        file_counts = {
            'config': len(list(model_dir.glob("*.json"))),
            'safetensors': len(list(model_dir.glob("*.safetensors"))),
            'pytorch': len(list(model_dir.glob("*.bin"))),
            'tokenizer': len(list(model_dir.glob("tokenizer*"))),
        }

        for file_type, count in file_counts.items():
            if count > 0:
                logger.info(f"  {file_type}: {count} file(s)")

        # Calculate total size
        total_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
        size_gb = total_size / (1024**3)
        logger.info(f"\nTotal size: {size_gb:.2f} GB")

    def verify_model(self, model_dir: Path) -> bool:
        """
        Verify model can be loaded with transformers.

        Args:
            model_dir: Path to model directory

        Returns:
            True if model loads successfully
        """
        logger.info(f"Verifying model at {model_dir}")

        try:
            from transformers import AutoConfig, AutoTokenizer

            # Load config
            logger.info("  Loading config...")
            config = AutoConfig.from_pretrained(model_dir)
            logger.info(f"  ✓ Config loaded: {config.model_type}")

            # Load tokenizer
            logger.info("  Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            logger.info(f"  ✓ Tokenizer loaded: {len(tokenizer)} tokens")

            # Don't load full model yet (too memory intensive)
            # Just verify the architecture is recognized
            logger.info(f"  Model architecture: {config.architectures}")

            logger.info("✓ Model verification successful")
            return True

        except Exception as e:
            logger.error(f"Model verification failed: {e}")
            return False


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def main():
    """Main entry point for standalone usage."""
    parser = argparse.ArgumentParser(
        description='Download models from HuggingFace',
        epilog="""
Examples:
  # Download SmolVLM2 2.2B model
  python scripts/download_model.py HuggingFaceTB/SmolVLM2-2.2B-Instruct

  # Download SmolVLM2 500M model
  python scripts/download_model.py HuggingFaceTB/SmolVLM2-500M-Video-Instruct

  # Force re-download
  python scripts/download_model.py HuggingFaceTB/SmolVLM2-2.2B-Instruct --force
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'model_id',
        help='HuggingFace model ID (e.g., HuggingFaceTB/SmolVLM2-2.2B-Instruct)'
    )
    parser.add_argument('--force', action='store_true', help='Force re-download')
    parser.add_argument('--no-verify', action='store_true', help='Skip verification')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    setup_logging(args.verbose)

    try:
        # Initialize config and downloader
        config = ConfigManager()
        downloader = ModelDownloader(config)

        # Download model
        model_dir = downloader.download_model(
            args.model_id,
            force_download=args.force,
            skip_if_exists=not args.force
        )

        # Verify model
        if not args.no_verify:
            if not downloader.verify_model(model_dir):
                logger.error("Model verification failed!")
                return 1

        logger.info(f"\n{'='*70}")
        logger.info("Model download complete!")
        logger.info(f"Location: {model_dir}")
        logger.info(f"{'='*70}")

        return 0

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
