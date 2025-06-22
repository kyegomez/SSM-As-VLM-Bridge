#!/usr/bin/env python3
"""
Comprehensive testing script for SSM-VLM Bridge Model
Tests all components and functions with detailed logging using loguru
"""

import torch
from loguru import logger
import time
from typing import Dict, Any

# Import the model components
from ssm_bridge.model import (
    VLMConfig,
    Swish,
    SwiGLU,
    MultiQueryAttention,
    CrossModalAttention,
    TransformerBlock,
    SSMLayer,
    EnhancedSSMBridge,
    EnhancedVisionTransformer,
    EnhancedVLM,
)


class ModelTester:
    """Comprehensive tester for all model components"""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.test_results = {}
        logger.info(
            f"Initializing ModelTester on device: {self.device}"
        )

    def log_test_result(
        self,
        test_name: str,
        success: bool,
        details: str = "",
        duration: float = 0.0,
    ):
        """Log test results with consistent formatting"""
        status = "✅ PASS" if success else "❌ FAIL"
        duration_str = f" ({duration:.3f}s)" if duration > 0 else ""
        logger.info(f"{status} {test_name}{duration_str}")
        if details:
            logger.debug(f"  Details: {details}")

        self.test_results[test_name] = {
            "success": success,
            "details": details,
            "duration": duration,
        }

    def test_swish_activation(self) -> bool:
        """Test Swish activation function"""
        logger.info("Testing Swish activation function...")
        start_time = time.time()

        try:
            swish = Swish()
            x = torch.randn(10, 20, device=self.device)

            # Test forward pass
            output = swish(x)

            # Check output shape
            assert (
                output.shape == x.shape
            ), f"Output shape {output.shape} != input shape {x.shape}"

            # Check numerical stability (no NaN or inf)
            assert not torch.isnan(
                output
            ).any(), "Output contains NaN values"
            assert not torch.isinf(
                output
            ).any(), "Output contains infinite values"

            # Check that output is in reasonable range
            assert torch.all(
                output >= -1.0
            ), "Output values too negative"
            assert torch.all(
                output <= 10.0
            ), "Output values too large"

            duration = time.time() - start_time
            self.log_test_result(
                "Swish Activation",
                True,
                f"Output shape: {output.shape}",
                duration,
            )
            return True

        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Swish Activation", False, str(e), duration
            )
            return False

    def test_swiglu(self) -> bool:
        """Test SwiGLU activation function"""
        logger.info("Testing SwiGLU activation function...")
        start_time = time.time()

        try:
            dim, hidden_dim = 256, 512
            swiglu = SwiGLU(dim, hidden_dim, dropout=0.1).to(
                self.device
            )
            x = torch.randn(4, 10, dim, device=self.device)

            # Test forward pass
            output = swiglu(x)

            # Check output shape
            assert (
                output.shape == x.shape
            ), f"Output shape {output.shape} != input shape {x.shape}"

            # Check numerical stability
            assert not torch.isnan(
                output
            ).any(), "Output contains NaN values"
            assert not torch.isinf(
                output
            ).any(), "Output contains infinite values"

            # Test with different input sizes
            x2 = torch.randn(2, 5, dim, device=self.device)
            output2 = swiglu(x2)
            assert (
                output2.shape == x2.shape
            ), "Failed with different batch size"

            duration = time.time() - start_time
            self.log_test_result(
                "SwiGLU",
                True,
                f"Input: {x.shape}, Output: {output.shape}",
                duration,
            )
            return True

        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("SwiGLU", False, str(e), duration)
            return False

    def test_multi_query_attention(self) -> bool:
        """Test Multi-Query Attention"""
        logger.info("Testing Multi-Query Attention...")
        start_time = time.time()

        try:
            dim, num_heads, kv_heads = 256, 8, 2
            mqa = MultiQueryAttention(
                dim, num_heads, kv_heads, dropout=0.1
            ).to(self.device)
            x = torch.randn(4, 20, dim, device=self.device)

            # Test forward pass without mask
            output = mqa(x)
            assert (
                output.shape == x.shape
            ), f"Output shape {output.shape} != input shape {x.shape}"

            # Test with causal mask
            mask = torch.tril(
                torch.ones(20, 20, device=self.device)
            ).unsqueeze(0)
            output_masked = mqa(x, mask)
            assert (
                output_masked.shape == x.shape
            ), "Masked output shape mismatch"

            # Check numerical stability
            assert not torch.isnan(
                output
            ).any(), "Output contains NaN values"
            assert not torch.isinf(
                output
            ).any(), "Output contains infinite values"

            duration = time.time() - start_time
            self.log_test_result(
                "Multi-Query Attention",
                True,
                f"Input: {x.shape}, Heads: {num_heads}/{kv_heads}",
                duration,
            )
            return True

        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Multi-Query Attention", False, str(e), duration
            )
            return False

    def test_cross_modal_attention(self) -> bool:
        """Test Cross-Modal Attention"""
        logger.info("Testing Cross-Modal Attention...")
        start_time = time.time()

        try:
            dim, num_heads = 256, 8
            cma = CrossModalAttention(dim, num_heads, dropout=0.1).to(
                self.device
            )

            query = torch.randn(4, 10, dim, device=self.device)
            key = torch.randn(4, 15, dim, device=self.device)
            value = torch.randn(4, 15, dim, device=self.device)

            # Test forward pass
            output = cma(query, key, value)
            assert (
                output.shape == query.shape
            ), f"Output shape {output.shape} != query shape {query.shape}"

            # Test with mask
            mask = torch.ones(4, 10, 15, device=self.device)
            output_masked = cma(query, key, value, mask)
            assert (
                output_masked.shape == query.shape
            ), "Masked output shape mismatch"

            # Check numerical stability
            assert not torch.isnan(
                output
            ).any(), "Output contains NaN values"
            assert not torch.isinf(
                output
            ).any(), "Output contains infinite values"

            duration = time.time() - start_time
            self.log_test_result(
                "Cross-Modal Attention",
                True,
                f"Query: {query.shape}, Key/Value: {key.shape}",
                duration,
            )
            return True

        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Cross-Modal Attention", False, str(e), duration
            )
            return False

    def test_transformer_block(self) -> bool:
        """Test Transformer Block"""
        logger.info("Testing Transformer Block...")
        start_time = time.time()

        try:
            dim, num_heads, kv_heads = 256, 8, 2
            ffn_dim = 1024
            block = TransformerBlock(
                dim, num_heads, kv_heads, ffn_dim, dropout=0.1
            ).to(self.device)
            x = torch.randn(4, 20, dim, device=self.device)

            # Test forward pass without mask
            output = block(x)
            assert (
                output.shape == x.shape
            ), f"Output shape {output.shape} != input shape {x.shape}"

            # Test with causal mask
            mask = torch.tril(
                torch.ones(20, 20, device=self.device)
            ).unsqueeze(0)
            output_masked = block(x, mask)
            assert (
                output_masked.shape == x.shape
            ), "Masked output shape mismatch"

            # Check numerical stability
            assert not torch.isnan(
                output
            ).any(), "Output contains NaN values"
            assert not torch.isinf(
                output
            ).any(), "Output contains infinite values"

            duration = time.time() - start_time
            self.log_test_result(
                "Transformer Block",
                True,
                f"Input: {x.shape}, Heads: {num_heads}/{kv_heads}",
                duration,
            )
            return True

        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Transformer Block", False, str(e), duration
            )
            return False

    def test_ssm_layer(self) -> bool:
        """Test SSM Layer"""
        logger.info("Testing SSM Layer...")
        start_time = time.time()

        try:
            hidden_dim, state_dim = 256, 64
            ssm = SSMLayer(hidden_dim, state_dim).to(self.device)
            x = torch.randn(4, 20, hidden_dim, device=self.device)

            # Test forward pass
            output = ssm(x)
            assert (
                output.shape == x.shape
            ), f"Output shape {output.shape} != input shape {x.shape}"

            # Check numerical stability
            assert not torch.isnan(
                output
            ).any(), "Output contains NaN values"
            assert not torch.isinf(
                output
            ).any(), "Output contains infinite values"

            # Test with different sequence lengths
            x2 = torch.randn(2, 10, hidden_dim, device=self.device)
            output2 = ssm(x2)
            assert (
                output2.shape == x2.shape
            ), "Failed with different sequence length"

            duration = time.time() - start_time
            self.log_test_result(
                "SSM Layer",
                True,
                f"Input: {x.shape}, State dim: {state_dim}",
                duration,
            )
            return True

        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("SSM Layer", False, str(e), duration)
            return False

    def test_enhanced_ssm_bridge(self) -> bool:
        """Test Enhanced SSM Bridge"""
        logger.info("Testing Enhanced SSM Bridge...")
        start_time = time.time()

        try:
            input_dim, state_dim, hidden_dim, output_dim = (
                768,
                64,
                256,
                768,
            )
            num_layers = 4
            bridge = EnhancedSSMBridge(
                input_dim,
                state_dim,
                hidden_dim,
                output_dim,
                num_layers,
                dropout=0.1,
            ).to(self.device)

            x = torch.randn(4, 20, input_dim, device=self.device)

            # Test forward pass
            output = bridge(x)
            assert output.shape == (
                4,
                20,
                output_dim,
            ), f"Output shape {output.shape} != expected {(4, 20, output_dim)}"

            # Check numerical stability
            assert not torch.isnan(
                output
            ).any(), "Output contains NaN values"
            assert not torch.isinf(
                output
            ).any(), "Output contains infinite values"

            # Test with different batch sizes
            x2 = torch.randn(2, 15, input_dim, device=self.device)
            output2 = bridge(x2)
            assert output2.shape == (
                2,
                15,
                output_dim,
            ), "Failed with different batch size"

            duration = time.time() - start_time
            self.log_test_result(
                "Enhanced SSM Bridge",
                True,
                f"Input: {x.shape}, Output: {output.shape}, Layers: {num_layers}",
                duration,
            )
            return True

        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Enhanced SSM Bridge", False, str(e), duration
            )
            return False

    def test_enhanced_vision_transformer(self) -> bool:
        """Test Enhanced Vision Transformer"""
        logger.info("Testing Enhanced Vision Transformer...")
        start_time = time.time()

        try:
            config = VLMConfig(
                img_size=224,
                patch_size=16,
                vision_embed_dim=768,
                vision_num_layers=4,  # Reduced for testing
                vision_num_heads=12,
                dropout=0.1,
            )

            vit = EnhancedVisionTransformer(config).to(self.device)
            x = torch.randn(4, 3, 224, 224, device=self.device)

            # Test forward pass
            output = vit(x)
            expected_patches = (
                224 // 16
            ) ** 2 + 1  # +1 for CLS token
            assert output.shape == (
                4,
                expected_patches,
                768,
            ), f"Output shape {output.shape} != expected {(4, expected_patches, 768)}"

            # Check numerical stability
            assert not torch.isnan(
                output
            ).any(), "Output contains NaN values"
            assert not torch.isinf(
                output
            ).any(), "Output contains infinite values"

            # Test with different batch sizes
            x2 = torch.randn(2, 3, 224, 224, device=self.device)
            output2 = vit(x2)
            assert output2.shape == (
                2,
                expected_patches,
                768,
            ), "Failed with different batch size"

            duration = time.time() - start_time
            self.log_test_result(
                "Enhanced Vision Transformer",
                True,
                f"Input: {x.shape}, Output: {output.shape}",
                duration,
            )
            return True

        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Enhanced Vision Transformer", False, str(e), duration
            )
            return False

    def test_enhanced_vlm_forward(self) -> bool:
        """Test Enhanced VLM forward pass"""
        logger.info("Testing Enhanced VLM forward pass...")
        start_time = time.time()

        try:
            config = VLMConfig(
                img_size=224,
                patch_size=16,
                vision_embed_dim=768,
                vision_num_layers=2,  # Reduced for testing
                vision_num_heads=12,
                vocab_size=32000,
                text_embed_dim=768,
                text_num_layers=2,  # Reduced for testing
                text_num_heads=12,
                text_kv_heads=1,
                max_seq_length=512,  # Reduced for testing
                ssm_state_dim=64,
                ssm_hidden_dim=256,
                ssm_num_layers=2,  # Reduced for testing
                ssm_dropout=0.1,
                cross_attn_layers=1,  # Reduced for testing
                cross_attn_heads=8,
                dropout=0.1,
                layer_norm_eps=1e-5,
            )

            model = EnhancedVLM(config).to(self.device)
            images = torch.randn(2, 3, 224, 224, device=self.device)
            tokens = torch.randint(
                0, 32000, (2, 10), device=self.device
            )
            targets = torch.randint(
                0, 32000, (2, 10), device=self.device
            )

            # Test forward pass with targets
            logits, loss = model(images, tokens, targets)
            assert logits.shape == (
                2,
                10,
                32000,
            ), f"Logits shape {logits.shape} != expected {(2, 10, 32000)}"
            assert isinstance(
                loss, torch.Tensor
            ), "Loss should be a tensor"
            assert loss.item() > 0, "Loss should be positive"

            # Test forward pass without targets
            logits_only = model(images, tokens)
            assert logits_only.shape == (
                2,
                10,
                32000,
            ), "Logits shape mismatch without targets"

            # Check numerical stability
            assert not torch.isnan(
                logits
            ).any(), "Logits contain NaN values"
            assert not torch.isinf(
                logits
            ).any(), "Logits contain infinite values"
            assert not torch.isnan(loss).any(), "Loss is NaN"
            assert not torch.isinf(loss).any(), "Loss is infinite"

            duration = time.time() - start_time
            self.log_test_result(
                "Enhanced VLM Forward",
                True,
                f"Logits: {logits.shape}, Loss: {loss.item():.4f}",
                duration,
            )
            return True

        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Enhanced VLM Forward", False, str(e), duration
            )
            return False

    def test_enhanced_vlm_generation(self) -> bool:
        """Test Enhanced VLM text generation"""
        logger.info("Testing Enhanced VLM text generation...")
        start_time = time.time()

        try:
            config = VLMConfig(
                img_size=224,
                patch_size=16,
                vision_embed_dim=768,
                vision_num_layers=2,
                vision_num_heads=12,
                vocab_size=32000,
                text_embed_dim=768,
                text_num_layers=2,
                text_num_heads=12,
                text_kv_heads=1,
                max_seq_length=512,
                ssm_state_dim=64,
                ssm_hidden_dim=256,
                ssm_num_layers=2,
                ssm_dropout=0.1,
                cross_attn_layers=1,
                cross_attn_heads=8,
                dropout=0.1,
                layer_norm_eps=1e-5,
            )

            model = EnhancedVLM(config).to(self.device)
            images = torch.randn(2, 3, 224, 224, device=self.device)

            # Test generation with different parameters
            generated = model.generate(
                images,
                max_length=5,  # Short for testing
                temperature=0.8,
                top_k=50,
                top_p=0.9,
            )

            assert (
                generated.shape[0] == 2
            ), f"Batch size mismatch: {generated.shape[0]} != 2"
            assert (
                generated.shape[1] <= 5
            ), f"Generated length {generated.shape[1]} > max_length 5"
            assert (
                generated.dtype == torch.long
            ), "Generated tokens should be long dtype"

            # Check that tokens are within vocabulary range
            assert torch.all(
                generated >= 0
            ), "Generated tokens contain negative values"
            assert torch.all(
                generated < 32000
            ), "Generated tokens exceed vocabulary size"

            duration = time.time() - start_time
            self.log_test_result(
                "Enhanced VLM Generation",
                True,
                f"Generated shape: {generated.shape}",
                duration,
            )
            return True

        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Enhanced VLM Generation", False, str(e), duration
            )
            return False

    def test_model_parameters(self) -> bool:
        """Test model parameter counting and gradients"""
        logger.info("Testing model parameters and gradients...")
        start_time = time.time()

        try:
            config = VLMConfig(
                img_size=224,
                patch_size=16,
                vision_embed_dim=768,
                vision_num_layers=2,
                vision_num_heads=12,
                vocab_size=32000,
                text_embed_dim=768,
                text_num_layers=2,
                text_num_heads=12,
                text_kv_heads=1,
                max_seq_length=512,
                ssm_state_dim=64,
                ssm_hidden_dim=256,
                ssm_num_layers=2,
                ssm_dropout=0.1,
                cross_attn_layers=1,
                cross_attn_heads=8,
                dropout=0.1,
                layer_norm_eps=1e-5,
            )

            model = EnhancedVLM(config).to(self.device)

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel()
                for p in model.parameters()
                if p.requires_grad
            )

            assert total_params > 0, "Model should have parameters"
            assert (
                trainable_params > 0
            ), "Model should have trainable parameters"
            assert (
                total_params == trainable_params
            ), "All parameters should be trainable"

            # Test gradient flow
            images = torch.randn(1, 3, 224, 224, device=self.device)
            tokens = torch.randint(
                0, 32000, (1, 5), device=self.device
            )
            targets = torch.randint(
                0, 32000, (1, 5), device=self.device
            )

            logits, loss = model(images, tokens, targets)
            loss.backward()

            # Check that gradients exist
            has_gradients = any(
                p.grad is not None for p in model.parameters()
            )
            assert (
                has_gradients
            ), "Model should have gradients after backward pass"

            duration = time.time() - start_time
            self.log_test_result(
                "Model Parameters",
                True,
                f"Total params: {total_params:,}, Trainable: {trainable_params:,}",
                duration,
            )
            return True

        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Model Parameters", False, str(e), duration
            )
            return False

    def test_memory_usage(self) -> bool:
        """Test memory usage and cleanup"""
        logger.info("Testing memory usage and cleanup...")
        start_time = time.time()

        try:
            config = VLMConfig(
                img_size=224,
                patch_size=16,
                vision_embed_dim=768,
                vision_num_layers=2,
                vision_num_heads=12,
                vocab_size=32000,
                text_embed_dim=768,
                text_num_layers=2,
                text_num_heads=12,
                text_kv_heads=1,
                max_seq_length=512,
                ssm_state_dim=64,
                ssm_hidden_dim=256,
                ssm_num_layers=2,
                ssm_dropout=0.1,
                cross_attn_layers=1,
                cross_attn_heads=8,
                dropout=0.1,
                layer_norm_eps=1e-5,
            )

            # Create model and test memory
            model = EnhancedVLM(config).to(self.device)
            images = torch.randn(2, 3, 224, 224, device=self.device)
            tokens = torch.randint(
                0, 32000, (2, 10), device=self.device
            )

            # Forward pass
            with torch.no_grad():
                logits = model(images, tokens)

            # Check memory cleanup
            del model, images, tokens, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            duration = time.time() - start_time
            self.log_test_result(
                "Memory Usage",
                True,
                "Memory cleanup successful",
                duration,
            )
            return True

        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Memory Usage", False, str(e), duration
            )
            return False

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return results"""
        logger.info("🚀 Starting comprehensive model testing...")
        logger.info("=" * 60)

        test_functions = [
            self.test_swish_activation,
            self.test_swiglu,
            self.test_multi_query_attention,
            self.test_cross_modal_attention,
            self.test_transformer_block,
            self.test_ssm_layer,
            self.test_enhanced_ssm_bridge,
            self.test_enhanced_vision_transformer,
            self.test_enhanced_vlm_forward,
            self.test_enhanced_vlm_generation,
            self.test_model_parameters,
            self.test_memory_usage,
        ]

        total_tests = len(test_functions)
        passed_tests = 0

        for test_func in test_functions:
            try:
                if test_func():
                    passed_tests += 1
            except Exception as e:
                logger.error(
                    f"Test {test_func.__name__} failed with exception: {e}"
                )
                self.log_test_result(
                    test_func.__name__, False, str(e)
                )

        # Summary
        logger.info("=" * 60)
        logger.info(
            f"📊 Test Summary: {passed_tests}/{total_tests} tests passed"
        )

        if passed_tests == total_tests:
            logger.success("🎉 All tests passed successfully!")
        else:
            logger.warning(
                f"⚠️  {total_tests - passed_tests} tests failed"
            )

        # Detailed results
        logger.info("\n📋 Detailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            duration = (
                f" ({result['duration']:.3f}s)"
                if result["duration"] > 0
                else ""
            )
            logger.info(f"  {status} {test_name}{duration}")
            if result["details"]:
                logger.debug(f"    Details: {result['details']}")

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": passed_tests / total_tests,
            "detailed_results": self.test_results,
        }


def main():
    """Main testing function"""
    # Configure loguru

    logger.info("🧪 SSM-VLM Bridge Model Testing Suite")
    logger.info("=" * 60)

    # Check device availability
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(
            f"🚀 CUDA available: {torch.cuda.get_device_name()}"
        )
    else:
        device = "cpu"
        logger.info("💻 Using CPU for testing")

    # Create tester and run tests
    tester = ModelTester(device)
    results = tester.run_all_tests()

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("🏁 Testing Complete!")
    logger.info(f"Success Rate: {results['success_rate']:.1%}")

    if results["success_rate"] == 1.0:
        logger.success("🎉 Perfect! All tests passed!")
    elif results["success_rate"] >= 0.8:
        logger.info("👍 Good! Most tests passed.")
    else:
        logger.warning(
            "⚠️  Several tests failed. Check the logs for details."
        )

    return results


if __name__ == "__main__":
    main()
