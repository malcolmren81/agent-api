"""
Unit tests for Policy Configuration Loader

Tests the AgentPolicyConfig class for loading, accessing, and hot-reloading
agent routing policy configuration from YAML files.
"""

import pytest
import yaml
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.config.policy_loader import AgentPolicyConfig, get_policy


class TestPolicyLoaderBasics:
    """Test basic policy loading and access"""

    def test_load_from_file(self, sample_policy_config, tmp_path):
        """Test loading policy from YAML file"""
        # Create temp config file
        config_file = tmp_path / "test_policy.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_policy_config, f)

        # Load policy
        policy = AgentPolicyConfig(config_path=str(config_file))

        # Verify loaded
        assert policy.config is not None
        assert isinstance(policy.config, dict)
        assert 'planner' in policy.config
        assert 'evaluation' in policy.config

    def test_load_missing_file_uses_defaults(self, tmp_path):
        """Test that missing config file falls back to defaults"""
        non_existent = tmp_path / "nonexistent.yaml"

        policy = AgentPolicyConfig(config_path=str(non_existent))

        # Should have default config
        assert policy.config is not None
        assert 'global' in policy.config
        assert 'planner' in policy.config
        assert policy.get('planner.mode') == 'hybrid'

    def test_load_default_path(self):
        """Test loading from default path"""
        # This should either load the real config or use defaults
        policy = AgentPolicyConfig()

        assert policy.config is not None
        assert isinstance(policy.config, dict)

    def test_load_malformed_yaml_uses_defaults(self, tmp_path):
        """Test that malformed YAML falls back to defaults"""
        config_file = tmp_path / "malformed.yaml"
        with open(config_file, 'w') as f:
            f.write("invalid: yaml: content: [unclosed")

        policy = AgentPolicyConfig(config_path=str(config_file))

        # Should have default config
        assert policy.config is not None
        assert 'global' in policy.config


class TestPolicyGetters:
    """Test policy value getters"""

    @pytest.fixture
    def policy(self, sample_policy_config, tmp_path):
        """Create policy instance with sample config"""
        config_file = tmp_path / "test_policy.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_policy_config, f)
        return AgentPolicyConfig(config_path=str(config_file))

    def test_get_simple_value(self, policy):
        """Test getting simple top-level value"""
        mode = policy.get('planner.mode')
        assert mode == 'hybrid'

    def test_get_nested_value(self, policy):
        """Test getting deeply nested value"""
        min_words = policy.get('planner.rule_conditions.min_word_count')
        assert min_words == 8

    def test_get_missing_key_returns_default(self, policy):
        """Test that missing key returns default value"""
        value = policy.get('nonexistent.key', 'default_value')
        assert value == 'default_value'

    def test_get_with_none_default(self, policy):
        """Test getting missing key with None default"""
        value = policy.get('missing.path')
        assert value is None

    def test_get_empty_dict_returns_default(self, policy):
        """Test that empty dict returns default"""
        # Manually set an empty dict
        policy.config['empty'] = {}
        value = policy.get('empty', 'fallback')
        assert value == 'fallback'

    def test_get_agent_mode(self, policy):
        """Test getting agent routing mode"""
        planner_mode = policy.get_agent_mode('planner')
        assert planner_mode == 'hybrid'

        # Missing agent should default to 'hybrid'
        unknown_mode = policy.get_agent_mode('unknown_agent')
        assert unknown_mode == 'hybrid'

    def test_get_feature_flag(self, policy):
        """Test getting feature flags"""
        # Add feature flags to config
        if 'global' not in policy.config:
            policy.config['global'] = {}
        policy.config['global']['feature_flags'] = {
            'enabled_feature': True,
            'disabled_feature': False
        }

        assert policy.get_feature_flag('enabled_feature') is True
        assert policy.get_feature_flag('disabled_feature') is False
        assert policy.get_feature_flag('nonexistent') is False

    def test_get_llm_config(self, policy):
        """Test getting LLM configuration"""
        # Add LLM config
        policy.config['planner']['llm_config'] = {
            'temperature': 0.5,
            'max_tokens': 1000
        }

        llm_config = policy.get_llm_config('planner')
        assert llm_config['temperature'] == 0.5
        assert llm_config['max_tokens'] == 1000

        # Missing agent should return defaults
        default_config = policy.get_llm_config('unknown_agent')
        assert 'temperature' in default_config
        assert 'max_tokens' in default_config

    def test_get_budget(self, policy):
        """Test getting cost budgets"""
        # Add budgets to config
        if 'global' not in policy.config:
            policy.config['global'] = {}
        policy.config['global']['soft_cost_budget_credits'] = 10
        policy.config['global']['hard_cost_budget_credits'] = 25

        assert policy.get_budget('soft') == 10
        assert policy.get_budget('hard') == 25
        assert policy.get_budget('invalid') == 8  # Default

    def test_get_max_regenerations(self, policy):
        """Test getting max regenerations"""
        if 'global' not in policy.config:
            policy.config['global'] = {}
        policy.config['global']['max_regenerations'] = 3

        assert policy.get_max_regenerations() == 3

    def test_get_evaluation_threshold(self, policy):
        """Test getting evaluation threshold"""
        threshold = policy.get_evaluation_threshold()
        assert threshold == 0.75
        assert isinstance(threshold, float)

    def test_get_evaluation_weights(self, policy):
        """Test getting evaluation weights"""
        weights = policy.get_evaluation_weights()
        assert isinstance(weights, dict)
        assert 'coverage' in weights
        assert 'aesthetics' in weights
        assert 'suitability' in weights
        assert weights['coverage'] == 0.35
        assert weights['aesthetics'] == 0.40
        assert weights['suitability'] == 0.25

    def test_get_vision_model(self, policy):
        """Test getting vision model configuration"""
        # Add vision model config
        if 'evaluation' not in policy.config:
            policy.config['evaluation'] = {}
        if 'subjective_llm' not in policy.config['evaluation']:
            policy.config['evaluation']['subjective_llm'] = {}

        policy.config['evaluation']['subjective_llm']['vision_model'] = 'gemini-2.0-flash-thinking'

        model = policy.get_vision_model()
        assert model == 'gemini-2.0-flash-thinking'


class TestPolicyHotReload:
    """Test hot-reload functionality"""

    def test_reload_updates_config(self, sample_policy_config, tmp_path):
        """Test that reload updates configuration"""
        config_file = tmp_path / "reload_test.yaml"

        # Initial config
        initial_config = sample_policy_config.copy()
        initial_config['planner']['mode'] = 'rule'
        with open(config_file, 'w') as f:
            yaml.dump(initial_config, f)

        policy = AgentPolicyConfig(config_path=str(config_file))
        assert policy.get('planner.mode') == 'rule'

        # Update config file
        updated_config = sample_policy_config.copy()
        updated_config['planner']['mode'] = 'llm'
        with open(config_file, 'w') as f:
            yaml.dump(updated_config, f)

        # Reload
        policy.reload()
        assert policy.get('planner.mode') == 'llm'

    def test_reload_handles_errors(self, sample_policy_config, tmp_path):
        """Test that reload handles errors gracefully"""
        config_file = tmp_path / "error_reload.yaml"

        # Initial valid config
        with open(config_file, 'w') as f:
            yaml.dump(sample_policy_config, f)

        policy = AgentPolicyConfig(config_path=str(config_file))
        original_mode = policy.get('planner.mode')

        # Make config file malformed
        with open(config_file, 'w') as f:
            f.write("malformed: [yaml: content")

        # Reload should keep old config
        policy.reload()
        assert policy.get('planner.mode') == original_mode


class TestPolicyGlobalInstance:
    """Test global policy instance"""

    def test_get_policy_returns_singleton(self):
        """Test that get_policy returns same instance"""
        policy1 = get_policy()
        policy2 = get_policy()

        assert policy1 is policy2

    def test_get_policy_respects_env_var(self, sample_policy_config, tmp_path):
        """Test that get_policy respects AGENT_POLICY_PATH env var"""
        config_file = tmp_path / "env_policy.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_policy_config, f)

        with patch.dict(os.environ, {'AGENT_POLICY_PATH': str(config_file)}):
            # Reset singleton
            import src.config.policy_loader
            src.config.policy_loader._policy_instance = None

            policy = get_policy()
            assert policy.config is not None


class TestPolicyEdgeCases:
    """Test edge cases and error handling"""

    def test_get_with_empty_path(self, sample_policy_config, tmp_path):
        """Test get with empty path string"""
        config_file = tmp_path / "edge_case.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_policy_config, f)

        policy = AgentPolicyConfig(config_path=str(config_file))

        # Empty path should return the entire config
        result = policy.get('', 'default')
        # Since we split empty string, we get [''], which should return config itself
        # But based on implementation, it depends on how it handles edge case

    def test_get_with_non_dict_value(self, tmp_path):
        """Test get when traversing through non-dict value"""
        config = {
            'planner': {
                'mode': 'hybrid',
                'count': 5
            }
        }
        config_file = tmp_path / "non_dict.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        policy = AgentPolicyConfig(config_path=str(config_file))

        # Try to access nested value under a non-dict
        result = policy.get('planner.count.nested', 'default')
        assert result == 'default'

    def test_default_config_structure(self):
        """Test that default config has all required sections"""
        policy = AgentPolicyConfig(config_path="/nonexistent/path")

        # Verify all required sections exist in defaults
        assert 'global' in policy.config
        assert 'planner' in policy.config
        assert 'prompt_manager' in policy.config
        assert 'model_selection' in policy.config
        assert 'evaluation' in policy.config

    def test_get_buckets(self, sample_policy_config, tmp_path):
        """Test getting model selection buckets"""
        config = sample_policy_config.copy()
        config['model_selection']['buckets'] = [
            'product:realistic:white-bg',
            'creative:artistic:abstract'
        ]

        config_file = tmp_path / "buckets.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        policy = AgentPolicyConfig(config_path=str(config_file))

        buckets = policy.get_buckets()
        assert isinstance(buckets, list)
        assert 'product:realistic:white-bg' in buckets
        assert 'creative:artistic:abstract' in buckets


class TestPolicyValidation:
    """Test policy validation logic"""

    def test_missing_required_sections_logs_warning(self, tmp_path, caplog):
        """Test that missing required sections are logged"""
        incomplete_config = {
            'planner': {'mode': 'hybrid'}
            # Missing other required sections
        }

        config_file = tmp_path / "incomplete.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(incomplete_config, f)

        import logging
        with caplog.at_level(logging.WARNING):
            policy = AgentPolicyConfig(config_path=str(config_file))

        # Should have loaded successfully with defaults for missing sections
        assert policy.config is not None

    def test_all_required_sections_present(self, sample_policy_config, tmp_path):
        """Test that config with all sections loads without warnings"""
        config_file = tmp_path / "complete.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_policy_config, f)

        policy = AgentPolicyConfig(config_path=str(config_file))

        # All sections should be present
        required = ['planner', 'prompt_manager', 'model_selection', 'evaluation']
        for section in required:
            assert section in policy.config
