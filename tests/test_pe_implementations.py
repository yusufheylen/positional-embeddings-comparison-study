"""Comprehensive tests for positional embedding implementations.

Tests:
- PoPE: PolarEmbedding, apply_polar_pos_emb, dimension doubling
- NoPE: nope factory, QK norm variants, position nullification
- Model factory: create_model with all PE types
- Data collators: Block diagonal attention masks
- DroPE callback: Mid-training PE switch
"""

import math
import pytest
from unittest.mock import MagicMock, patch

# Skip tests if dependencies not available
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
import torch.nn as nn


# ==============================================================================
# PoPE Tests
# ==============================================================================

class TestPolarEmbedding:
    """Test PolarEmbedding module."""

    def test_init(self):
        """Test PolarEmbedding initialization."""
        from src.models.embeddings.pope import PolarEmbedding

        dim, num_heads = 64, 8
        emb = PolarEmbedding(dim=dim, num_heads=num_heads)

        assert emb.inv_freq.shape == (dim,)
        assert emb.learned_bias.shape == (num_heads, 1, dim)
        # Bias should be initialized to zero
        assert torch.allclose(emb.learned_bias, torch.zeros_like(emb.learned_bias))

    def test_forward_shapes(self):
        """Test PolarEmbedding forward pass shapes."""
        from src.models.embeddings.pope import PolarEmbedding

        dim, num_heads, seq_len, batch = 64, 8, 128, 2
        emb = PolarEmbedding(dim=dim, num_heads=num_heads)

        positions = torch.arange(seq_len)
        freqs, bias = emb(positions)

        assert freqs.shape == (1, seq_len, dim)
        assert bias.shape == (num_heads, 1, dim)

        # Test with batched positions
        batch_positions = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
        freqs, bias = emb(batch_positions)
        assert freqs.shape == (batch, seq_len, dim)

    def test_bias_clamping(self):
        """Test that learned bias is clamped to [-2pi, 0]."""
        from src.models.embeddings.pope import PolarEmbedding

        emb = PolarEmbedding(dim=32, num_heads=4)

        # Set bias outside valid range
        with torch.no_grad():
            emb.learned_bias.fill_(10.0)

        positions = torch.arange(10)
        _, bias = emb(positions)

        # Should be clamped to 0
        assert torch.all(bias <= 0)

        # Set to negative beyond range
        with torch.no_grad():
            emb.learned_bias.fill_(-100.0)

        _, bias = emb(positions)
        # Should be clamped to -2*pi
        assert torch.all(bias >= -2 * math.pi)

    def test_frequency_computation(self):
        """Test inverse frequency computation follows expected pattern."""
        from src.models.embeddings.pope import PolarEmbedding

        dim, base = 8, 10000.0
        emb = PolarEmbedding(dim=dim, num_heads=1, base=base)

        # Expected: inv_freq[c] = 1 / (base^(c/dim))
        expected = 1.0 / (base ** (torch.arange(dim).float() / dim))
        assert torch.allclose(emb.inv_freq, expected)


class TestApplyPolarPosEmb:
    """Test apply_polar_pos_emb function."""

    def test_output_shape_doubles_dim(self):
        """Test that output dimension is doubled."""
        from src.models.embeddings.pope import apply_polar_pos_emb

        batch, seq, dim = 2, 10, 64
        t = torch.randn(batch, seq, dim)
        freqs = torch.randn(batch, seq, dim)

        out = apply_polar_pos_emb(t, freqs)

        assert out.shape == (batch, seq, 2 * dim)

    def test_softplus_makes_magnitude_positive(self):
        """Test that softplus ensures non-negative magnitudes."""
        from src.models.embeddings.pope import apply_polar_pos_emb

        # Create negative input
        t = torch.full((1, 1, 4), -10.0)
        freqs = torch.zeros(1, 1, 4)

        out = apply_polar_pos_emb(t, freqs)

        # First half is t * cos(0) = softplus(t) * 1
        # Should be positive due to softplus
        assert torch.all(out[..., :4] > 0)

    def test_polar_to_cartesian(self):
        """Test conversion to Cartesian coordinates."""
        from src.models.embeddings.pope import apply_polar_pos_emb
        import torch.nn.functional as F

        # Create simple test case
        t = torch.ones(1, 1, 2)  # magnitude 1 after softplus
        freqs = torch.tensor([[[0.0, math.pi / 2]]])  # angles 0 and 90 degrees

        out = apply_polar_pos_emb(t, freqs)

        mag = F.softplus(torch.tensor(1.0)).item()

        # For angle 0: [mag * cos(0), mag * sin(0)] = [mag, 0]
        # For angle pi/2: [mag * cos(pi/2), mag * sin(pi/2)] = [0, mag]
        # Output: [cos part for all dims, sin part for all dims]
        # So: [mag*cos(0), mag*cos(pi/2), mag*sin(0), mag*sin(pi/2)]
        assert torch.isclose(out[0, 0, 0], torch.tensor(mag), rtol=1e-4)
        assert torch.isclose(out[0, 0, 1], torch.tensor(0.0), atol=1e-4)
        assert torch.isclose(out[0, 0, 2], torch.tensor(0.0), atol=1e-4)
        assert torch.isclose(out[0, 0, 3], torch.tensor(mag), rtol=1e-4)


# ==============================================================================
# NoPE Tests
# ==============================================================================

class TestNoPEFactory:
    """Test NoPE factory function and attention."""

    def test_nope_factory_creates_class(self):
        """Test that nope factory creates a proper class."""
        from src.models.embeddings.nope import nope
        from transformers.models.llama.modeling_llama import LlamaAttention

        NoPEAttention = nope(LlamaAttention)

        assert NoPEAttention.__name__ == "NoPELlamaAttention"
        assert issubclass(NoPEAttention, LlamaAttention)

    def test_nope_nullifies_position_embeddings(self):
        """Test that NoPE sets cos=1, sin=0."""
        from src.models.embeddings.nope import nope
        import torch.nn as nn

        # Create a mock attention class
        class MockAttention(nn.Module):
            def forward(self, hidden_states, position_embeddings=None, **kwargs):
                # Store what position_embeddings we receive
                self.received_position_embeddings = position_embeddings
                return hidden_states, None

        NoPEMock = nope(MockAttention)
        attn = NoPEMock()

        # Call with non-trivial position embeddings
        hidden = torch.randn(1, 4, 64)
        cos = torch.randn(1, 1, 4, 8)  # Random cos values
        sin = torch.randn(1, 1, 4, 8)  # Random sin values

        attn(hidden, position_embeddings=(cos, sin))

        # Check that position_embeddings were nullified
        received_cos, received_sin = attn.received_position_embeddings
        assert torch.allclose(received_cos, torch.ones_like(cos))
        assert torch.allclose(received_sin, torch.zeros_like(sin))


class TestQKNormNoPE:
    """Test QK normalization variants."""

    def test_qk_norm_adds_norm_layers(self):
        """Test that QK norm variant has normalization layers."""
        from src.models.embeddings.nope import QKNormNoPELlamaAttention
        from transformers import LlamaConfig

        config = LlamaConfig(
            hidden_size=256,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=512,
            rms_norm_eps=1e-6,
        )

        attn = QKNormNoPELlamaAttention(config, layer_idx=0)

        assert hasattr(attn, "q_norm")
        assert hasattr(attn, "k_norm")
        assert isinstance(attn.q_norm, nn.RMSNorm)
        assert isinstance(attn.k_norm, nn.RMSNorm)


class TestNopeAttentionVariants:
    """Test pre-built NoPE attention variants exist."""

    def test_variants_exist(self):
        """Test that all variants are defined."""
        from src.models.embeddings.nope import (
            NoPELlamaAttention,
            QKNormNoPELlamaAttention,
            QNormNoPELlamaAttention,
            KNormNoPELlamaAttention,
            NOPE_ATTENTION_VARIANTS,
        )
        from transformers.models.llama.modeling_llama import LlamaAttention

        assert LlamaAttention in NOPE_ATTENTION_VARIANTS
        variants = NOPE_ATTENTION_VARIANTS[LlamaAttention]

        assert "nope" in variants
        assert "qk_norm_nope" in variants
        assert "q_norm_nope" in variants
        assert "k_norm_nope" in variants


# ==============================================================================
# Model Factory Tests
# ==============================================================================

class TestModelFactory:
    """Test create_model and related functions."""

    def test_get_best_attn_implementation(self):
        """Test attention implementation detection."""
        from src.models.base import get_best_attn_implementation

        impl = get_best_attn_implementation()
        assert impl in ("flash_attention_2", "sdpa", "eager")

    def test_get_model_config_rope(self):
        """Test config creation for RoPE."""
        from src.models.base import get_model_config

        # Use a small model for testing
        config = get_model_config(
            "HuggingFaceTB/SmolLM2-135M",
            pe_type="rope",
            trust_remote_code=True,
        )

        assert config is not None
        # RoPE should not be configured as YaRN (Transformers 5.x may include default RoPE metadata here)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None:
            rope_type = rope_scaling.get("type", rope_scaling.get("rope_type"))
            assert rope_type != "yarn"

    def test_get_model_config_yarn(self):
        """Test config creation for YaRN."""
        from src.models.base import get_model_config

        config = get_model_config(
            "HuggingFaceTB/SmolLM2-135M",
            pe_type="yarn",
            yarn_factor=2.0,
            trust_remote_code=True,
        )

        assert config.rope_scaling is not None
        assert config.rope_scaling["type"] == "yarn"
        assert config.rope_scaling["factor"] == 2.0

    def test_model_arch_map_exists(self):
        """Test that MODEL_ARCH_MAP is properly defined."""
        from src.models.base import MODEL_ARCH_MAP
        from transformers.models.llama.modeling_llama import LlamaForCausalLM

        assert LlamaForCausalLM in MODEL_ARCH_MAP


# ==============================================================================
# Data Collator Tests
# ==============================================================================

class TestBlockDiagFromEOSCollator:
    """Test BlockDiagFromEOSCollator."""

    def test_basic_collation(self):
        """Test basic batch collation."""
        from src.data.dataset import BlockDiagFromEOSCollator
        from transformers import AutoTokenizer

        tokenizer = MagicMock()
        tokenizer.eos_token_id = 2

        collator = BlockDiagFromEOSCollator(tokenizer)

        # Create features with EOS tokens marking document boundaries
        features = [
            {
                "input_ids": [1, 5, 6, 2, 7, 8, 2],  # Two docs separated by EOS
                "attention_mask": [1, 1, 1, 1, 1, 1, 1],
            }
        ]

        batch = collator(features)

        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch

        # Attention mask should be 4D
        assert batch["attention_mask"].dim() == 4

    def test_segment_isolation(self):
        """Test that segments are isolated in attention."""
        from src.data.dataset import BlockDiagFromEOSCollator

        tokenizer = MagicMock()
        tokenizer.eos_token_id = 2

        collator = BlockDiagFromEOSCollator(tokenizer)

        # Single sequence: [doc1 tokens, EOS, doc2 tokens]
        # Positions:       [0, 1, 2, 3, 4, 5]
        features = [
            {
                "input_ids": [1, 5, 2, 7, 8, 9],  # EOS at position 2
                "attention_mask": [1, 1, 1, 1, 1, 1],
            }
        ]

        batch = collator(features)
        attn_mask = batch["attention_mask"][0, 0]  # (seq, seq)

        # Token 3 (first of doc2) should NOT attend to tokens 0-2 (doc1)
        # Check if position 3 can attend to position 1 (should be -inf or very negative)
        assert attn_mask[3, 1] < -1e9  # Very negative = no attention

        # Token 4 should attend to token 3 (same segment)
        assert attn_mask[4, 3] == 0.0  # 0 means attend


class TestBlockDiagFA2Collator:
    """Test BlockDiagFA2Collator for Flash Attention 2."""

    def test_cu_seqlens_format(self):
        """Test that output is in FA2 format with cu_seqlens."""
        from src.data.dataset import BlockDiagFA2Collator

        tokenizer = MagicMock()
        tokenizer.eos_token_id = 2

        collator = BlockDiagFA2Collator(tokenizer)

        features = [
            {"input_ids": [1, 5, 2, 7, 8, 2]},  # Two segments: [1,5,2] and [7,8,2]
        ]

        batch = collator(features)

        assert "cu_seq_lens_q" in batch
        assert "cu_seq_lens_k" in batch
        assert "max_length_q" in batch
        assert "max_length_k" in batch
        assert "position_ids" in batch

        # cu_seqlens should start with 0 and be monotonically increasing
        cu = batch["cu_seq_lens_q"]
        assert cu[0] == 0
        assert torch.all(cu[1:] > cu[:-1])

    def test_position_ids_reset_per_segment(self):
        """Test that position IDs reset for each segment."""
        from src.data.dataset import BlockDiagFA2Collator

        tokenizer = MagicMock()
        tokenizer.eos_token_id = 2

        collator = BlockDiagFA2Collator(tokenizer)

        features = [
            {"input_ids": [1, 5, 2, 7, 8, 9, 2]},  # Seg1: [1,5,2], Seg2: [7,8,9,2]
        ]

        batch = collator(features)
        pos_ids = batch["position_ids"][0].tolist()

        # First segment: positions 0, 1, 2
        # Second segment: positions 0, 1, 2, 3
        # So we expect: [0, 1, 2, 0, 1, 2, 3]
        assert pos_ids == [0, 1, 2, 0, 1, 2, 3]


class TestGetDataCollator:
    """Test get_data_collator factory function."""

    def test_returns_correct_collator_for_flash(self):
        """Test FA2 collator is returned for flash attention."""
        from src.data.dataset import get_data_collator, BlockDiagFA2Collator

        tokenizer = MagicMock()
        tokenizer.eos_token_id = 2

        collator = get_data_collator(tokenizer, attn_implementation="flash_attention_2")
        assert isinstance(collator, BlockDiagFA2Collator)

    def test_returns_correct_collator_for_sdpa(self):
        """Test SDPA collator is returned for sdpa attention."""
        from src.data.dataset import get_data_collator, BlockDiagFromEOSCollator

        tokenizer = MagicMock()
        tokenizer.eos_token_id = 2

        collator = get_data_collator(tokenizer, attn_implementation="sdpa")
        assert isinstance(collator, BlockDiagFromEOSCollator)

    def test_returns_default_when_no_masking(self):
        """Test default collator when masking disabled."""
        from src.data.dataset import get_data_collator
        from transformers import default_data_collator

        tokenizer = MagicMock()

        collator = get_data_collator(tokenizer, mask_past_sequences=False)
        assert collator == default_data_collator


# ==============================================================================
# DroPE Callback Tests
# ==============================================================================

class TestDropeCallback:
    """Test DroPE callback for mid-training PE switch."""

    def test_callback_initialization(self):
        """Test DroPE callback initializes correctly."""
        from src.training.drope_callback import DroPECallback

        callback = DroPECallback(switch_step=50000)

        assert callback.switch_step == 50000
        assert callback.has_switched is False

    def test_switch_triggers_at_correct_step(self):
        """Test that switch happens at the right step."""
        from src.training.drope_callback import DroPECallback

        callback = DroPECallback(switch_step=100)

        # Mock trainer state
        state = MagicMock()
        state.global_step = 99

        # Create mock control
        control = MagicMock()

        # Before switch step - should not switch
        callback.on_step_begin(None, state, control, model=MagicMock())
        assert callback.has_switched is False

        # At switch step - should switch
        state.global_step = 100
        with patch.object(callback, "_perform_switch") as mock_switch:
            callback.on_step_begin(None, state, control, model=MagicMock())
            mock_switch.assert_called_once()


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.mark.slow
    def test_polar_embedding_gradients_flow(self):
        """Test that gradients flow through PoPE."""
        from src.models.embeddings.pope import PolarEmbedding, apply_polar_pos_emb

        emb = PolarEmbedding(dim=32, num_heads=4)

        positions = torch.arange(10)
        freqs, bias = emb(positions)

        # Create input requiring grad
        t = torch.randn(1, 4, 10, 32, requires_grad=True)  # (batch, heads, seq, dim)
        freqs_expanded = freqs.unsqueeze(1)  # (batch, 1, seq, dim) for broadcast

        # Use freqs with bias (like key computation) to ensure bias gets gradients
        key_freqs = freqs_expanded + bias.unsqueeze(0)  # (batch, heads, seq, dim)
        out = apply_polar_pos_emb(t, key_freqs)
        loss = out.sum()
        loss.backward()

        # Check gradients exist
        assert t.grad is not None
        assert emb.learned_bias.grad is not None


# ==============================================================================
# Run tests
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
