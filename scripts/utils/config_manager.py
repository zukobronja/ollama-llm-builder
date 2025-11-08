#!/usr/bin/env python3
"""
Configuration Manager for Ollama LLM Builder

Handles loading and managing hardware profiles and build configurations.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """Manages configuration files for model building."""

    def __init__(self, config_dir: str = None):
        """
        Initialize ConfigManager.

        Args:
            config_dir: Path to config directory. Defaults to project root/config
        """
        if config_dir is None:
            # Assume script is in scripts/utils/, go up two levels
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "config"

        self.config_dir = Path(config_dir)
        self.hardware_profiles_path = self.config_dir / "hardware_profiles.yaml"
        self.build_config_path = self.config_dir / "build_config.yaml"

        self.hardware_profiles = self._load_yaml(self.hardware_profiles_path)
        self.build_config = self._load_yaml(self.build_config_path)

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def get_active_profile(self) -> Dict[str, Any]:
        """Get the currently active hardware profile."""
        profile_name = self.build_config.get('active_profile', 'laptop_rtx4070')
        return self.get_profile(profile_name)

    def get_profile(self, profile_name: str) -> Dict[str, Any]:
        """Get a specific hardware profile by name."""
        profiles = self.hardware_profiles.get('profiles', {})
        if profile_name not in profiles:
            available = list(profiles.keys())
            raise ValueError(
                f"Profile '{profile_name}' not found. "
                f"Available profiles: {available}"
            )
        return profiles[profile_name]

    def list_profiles(self) -> list:
        """List all available hardware profiles."""
        return list(self.hardware_profiles.get('profiles', {}).keys())

    def get_quantization_info(self, quant_type: str) -> Dict[str, Any]:
        """Get information about a specific quantization type."""
        quant_info = self.hardware_profiles.get('quantization_info', {})
        if quant_type not in quant_info:
            available = list(quant_info.keys())
            raise ValueError(
                f"Quantization type '{quant_type}' not found. "
                f"Available types: {available}"
            )
        return quant_info[quant_type]


    def get_ollama_params(self, profile_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get Ollama parameters for a specific profile or active profile.

        Args:
            profile_name: Name of the profile. Uses active profile if None.

        Returns:
            Dictionary of Ollama parameters
        """
        if profile_name is None:
            profile = self.get_active_profile()
        else:
            profile = self.get_profile(profile_name)

        return profile.get('ollama_params', {})

    def get_recommended_quantization(self, profile_name: Optional[str] = None) -> list:
        """
        Get recommended quantization levels for a profile.

        Args:
            profile_name: Name of the profile. Uses active profile if None.

        Returns:
            List of recommended quantization types
        """
        if profile_name is None:
            profile = self.get_active_profile()
        else:
            profile = self.get_profile(profile_name)

        return profile.get('recommended_quantization', ['Q4_K_M'])

    def get_build_config(self) -> Dict[str, Any]:
        """Get the full build configuration."""
        return self.build_config

    def get_paths(self) -> Dict[str, str]:
        """Get configured paths."""
        return self.build_config.get('paths', {})

    def estimate_vram_usage(self, model_size_gb: float, quant_type: str) -> float:
        """
        Estimate VRAM usage for a model with specific quantization.

        Args:
            model_size_gb: Base model size in GB (FP16)
            quant_type: Quantization type (e.g., 'Q4_K_M')

        Returns:
            Estimated VRAM usage in GB
        """
        quant_info = self.get_quantization_info(quant_type)
        multiplier = quant_info.get('vram_multiplier', 0.25)
        estimated = model_size_gb * multiplier

        # Add overhead for context and attention (rough estimate: +1-2GB)
        overhead = 1.5

        return estimated + overhead


    def print_profile_info(self, profile_name: Optional[str] = None):
        """Print detailed information about a profile."""
        if profile_name is None:
            profile_name = self.build_config.get('active_profile')
            print(f"Active Profile: {profile_name}")
            print("=" * 60)

        profile = self.get_profile(profile_name)

        print(f"Name: {profile.get('name')}")
        print(f"Description: {profile.get('description')}")
        print("\nSpecs:")
        for key, value in profile.get('specs', {}).items():
            print(f"  {key}: {value}")

        print("\nRecommended Quantization:")
        for quant in profile.get('recommended_quantization', []):
            info = self.get_quantization_info(quant)
            print(f"  {quant}: {info.get('description')} - {info.get('use_case')}")

        print("\nOllama Parameters:")
        for key, value in profile.get('ollama_params', {}).items():
            print(f"  {key}: {value}")

        if 'notes' in profile:
            print(f"\nNotes: {profile.get('notes')}")


def main():
    """Demo of ConfigManager functionality."""
    config = ConfigManager()

    print("Available Hardware Profiles:")
    print("-" * 60)
    for profile_name in config.list_profiles():
        profile = config.get_profile(profile_name)
        print(f"  {profile_name}: {profile.get('description')}")

    print("\n")
    config.print_profile_info()

    print("\n")
    print("Compatibility Check for SmolVLM2:")
    print("-" * 60)
    compat = config.check_hardware_compatibility('smolvlm2_instruct')
    print(f"Compatible: {compat['compatible']}")
    if compat['warnings']:
        print("Warnings:")
        for warning in compat['warnings']:
            print(f"  - {warning}")
    if compat['recommendations']:
        print("Recommendations:")
        for rec in compat['recommendations']:
            print(f"  - {rec}")


if __name__ == "__main__":
    main()
