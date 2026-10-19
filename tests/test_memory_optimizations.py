"""Check that low-memory training preserves outputs and gradients."""
from unittest.mock import patch

import torch
from transformers import ConvNextConfig, ConvNextModel
from torch.utils.checkpoint import checkpoint

from src.models.legacy_layers import GuideDecoderLayer
from src.models.legacy_model import VisionModel


def test_unused_attention_weights_preserve_output_and_gradients():
    torch.manual_seed(7)
    layer = GuideDecoderLayer(32, 12).eval()
    image = torch.randn(1, 49, 32, requires_grad=True)
    text = torch.randn(1, 24, 768, requires_grad=True)
    optimized = layer(image, text)
    optimized.sum().backward()
    gradients = [p.grad.clone() for p in layer.parameters()]
    layer.zero_grad(set_to_none=True)
    original_self = layer.self_attn.forward
    original_cross = layer.cross_attn.forward

    def with_weights(method):
        def forward(*args, **kwargs):
            kwargs['need_weights'] = True
            return method(*args, **kwargs)
        return forward

    with patch.object(layer.self_attn, 'forward', with_weights(original_self)), \
         patch.object(layer.cross_attn, 'forward', with_weights(original_cross)):
        reference = layer(image, text)
        reference.sum().backward()
    torch.testing.assert_close(optimized, reference, atol=2e-5, rtol=2e-5)
    for actual, parameter in zip(gradients, layer.parameters()):
        torch.testing.assert_close(actual, parameter.grad, atol=2e-4, rtol=2e-4)


def test_encoder_checkpointing_preserves_output_and_gradients():
    torch.manual_seed(7)
    config = ConvNextConfig(depths=[1, 1, 1, 1], hidden_sizes=[8, 16, 32, 768])
    with patch('src.models.legacy_model.AutoModel.from_pretrained',
               return_value=ConvNextModel(config)):
        encoder = VisionModel('test-only', 16).train()
    image = torch.randn(1, 3, 32, 32)
    reference = encoder(image)['project']
    reference.sum().backward()
    gradients = {name: p.grad.clone() for name, p in encoder.named_parameters()
                 if p.grad is not None}
    encoder.zero_grad(set_to_none=True)
    encoder.memory_checkpointing = True
    with patch('torch.utils.checkpoint.checkpoint', wraps=checkpoint) as wrapped:
        optimized = encoder(image)['project']
        optimized.sum().backward()
        wrapped.assert_called_once()
    torch.testing.assert_close(optimized, reference)
    for name, parameter in encoder.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name])
