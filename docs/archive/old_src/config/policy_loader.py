"""
Agent Routing Policy Configuration Loader

Loads and manages the agent routing policy from YAML configuration file.
Supports hot-reloading for runtime configuration updates.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class AgentPolicyConfig:
    """
    Configuration manager for agent routing policies.

    Loads configuration from YAML file and provides convenient access
    to nested configuration values with fallback defaults.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize policy configuration.

        Args:
            config_path: Path to YAML config file. If None, uses default path.
        """
        if config_path is None:
            # Default to config/agent_routing_policy.yaml in project root
            base_path = Path(__file__).parent.parent.parent
            config_path = base_path / "config" / "agent_routing_policy.yaml"

        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self._load()

        logger.info(f"Loaded agent routing policy from {self.config_path}")

    def _load(self) -> None:
        """Load configuration from YAML file."""
        try:
            if not self.config_path.exists():
                logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
                self.config = self._get_default_config()
                return

            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)

            # Validate required sections
            required_sections = ['global', 'planner', 'prompt_manager', 'model_selection', 'evaluation']
            for section in required_sections:
                if section not in self.config:
                    logger.warning(f"Missing config section: {section}. Using defaults for this section.")

        except Exception as e:
            logger.error(f"Error loading config: {e}. Using defaults.")
            self.config = self._get_default_config()

    def reload(self) -> None:
        """Hot-reload configuration from file."""
        logger.info("Reloading agent routing policy...")
        old_config = self.config.copy()

        try:
            self._load()
            logger.info("Successfully reloaded configuration")
        except Exception as e:
            logger.error(f"Failed to reload config: {e}. Keeping previous configuration.")
            self.config = old_config

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get nested configuration value using dot notation.

        Args:
            path: Dot-separated path to config value (e.g., 'planner.mode')
            default: Default value if path not found

        Returns:
            Configuration value or default

        Examples:
            >>> policy.get('planner.mode')
            'hybrid'
            >>> policy.get('global.soft_cost_budget_credits', 10)
            8
        """
        keys = path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value if value != {} else default

    def get_feature_flag(self, flag_name: str) -> bool:
        """
        Get feature flag value.

        Args:
            flag_name: Name of the feature flag

        Returns:
            Boolean flag value, defaults to False if not found
        """
        return self.get(f'global.feature_flags.{flag_name}', False)

    def get_llm_config(self, agent_name: str) -> Dict[str, Any]:
        """
        Get LLM configuration for a specific agent.

        Args:
            agent_name: Name of the agent (planner, prompt_manager, etc.)

        Returns:
            Dict with LLM config (temperature, max_tokens, etc.)
        """
        return self.get(f'{agent_name}.llm_config', {
            'temperature': 0.3,
            'max_tokens': 500
        })

    def get_agent_mode(self, agent_name: str) -> str:
        """
        Get routing mode for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Routing mode: 'rule', 'llm', or 'hybrid'
        """
        return self.get(f'{agent_name}.mode', 'hybrid')

    def get_buckets(self) -> List[str]:
        """Get list of use-case buckets for model selection."""
        return self.get('model_selection.buckets', [
            'product:realistic:white-bg',
            'product:realistic:lifestyle',
            'product:artistic:lifestyle',
            'creative:artistic:abstract',
            'creative:realistic:scene'
        ])

    def get_budget(self, budget_type: str = 'soft') -> int:
        """
        Get cost budget.

        Args:
            budget_type: 'soft' or 'hard'

        Returns:
            Budget in credits
        """
        if budget_type == 'soft':
            return self.get('global.soft_cost_budget_credits', 8)
        elif budget_type == 'hard':
            return self.get('global.hard_cost_budget_credits', 20)
        else:
            return 8

    def get_max_regenerations(self) -> int:
        """Get maximum number of regeneration attempts."""
        return self.get('global.max_regenerations', 1)

    def get_evaluation_threshold(self) -> float:
        """Get acceptance threshold for evaluation."""
        return self.get('evaluation.acceptance_threshold', 0.75)

    def get_evaluation_weights(self) -> Dict[str, float]:
        """Get weights for combining evaluation scores."""
        return self.get('evaluation.combined_weights', {
            'coverage': 0.35,
            'aesthetics': 0.40,
            'suitability': 0.25
        })

    def get_vision_model(self) -> str:
        """Get vision model for evaluation."""
        return self.get('evaluation.subjective_llm.vision_model', 'gemini-2.0-flash')

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration fallback.

        Returns:
            Default config dictionary
        """
        return {
            'version': '1.0',
            'global': {
                'soft_cost_budget_credits': 8,
                'hard_cost_budget_credits': 20,
                'max_regenerations': 1,
                'feature_flags': {
                    'planner_llm_enabled': True,
                    'prompt_mgr_llm_enabled': True,
                    'model_selector_bandit_enabled': True,
                    'evaluation_vision_enabled': True
                }
            },
            'planner': {
                'mode': 'hybrid',
                'rule_conditions': {
                    'min_word_count': 8,
                    'max_objects': 1,
                    'novelty_threshold': 0.35
                }
            },
            'prompt_manager': {
                'mode': 'hybrid',
                'primary': 'database',
                'template_confidence_threshold': 0.80
            },
            'model_selection': {
                'strategy': 'ucb1',
                'exploration': {
                    'min_trials_per_model': 1,
                    'decay_type': 'ema',
                    'decay_alpha': 0.031
                }
            },
            'evaluation': {
                'mode': 'hybrid',
                'objective_checks': {
                    'min_resolution': [1024, 1024],
                    'min_coverage': 0.70,
                    'background_whiteness': 0.92
                },
                'acceptance_threshold': 0.75,
                'combined_weights': {
                    'coverage': 0.35,
                    'aesthetics': 0.40,
                    'suitability': 0.25
                }
            }
        }


# Global singleton instance
# Usage: from config.policy_loader import policy
_policy_instance: Optional[AgentPolicyConfig] = None


def get_policy() -> AgentPolicyConfig:
    """Get global policy configuration instance."""
    global _policy_instance
    if _policy_instance is None:
        # Check for environment variable override
        config_path = os.getenv('AGENT_POLICY_PATH')
        _policy_instance = AgentPolicyConfig(config_path)
    return _policy_instance


# Convenience export
policy = get_policy()
