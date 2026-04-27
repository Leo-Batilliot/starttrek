import unittest

import torch
import torch.nn as nn

from src.networks.mlp import build_mlp


class TestMLPShapes(unittest.TestCase):

    def test_forward_single_sample(self):
        net = build_mlp(8, 4, [128, 128], "relu")
        out = net(torch.zeros(1, 8))
        self.assertEqual(out.shape, torch.Size([1, 4]))

    def test_forward_batch(self):
        net = build_mlp(8, 4, [128, 128], "relu")
        out = net(torch.zeros(32, 8))
        self.assertEqual(out.shape, torch.Size([32, 4]))

    def test_param_count_canonical(self):
        net = build_mlp(8, 4, [128, 128], "relu")
        expected = 8 * 128 + 128 + 128 * 128 + 128 + 128 * 4 + 4
        actual = sum(p.numel() for p in net.parameters())
        self.assertEqual(actual, expected)

    def test_returns_sequential(self):
        net = build_mlp(8, 4, [64], "relu")
        self.assertIsInstance(net, nn.Sequential)


class TestMLPArchitecture(unittest.TestCase):

    def test_empty_hidden_sizes_is_single_linear(self):
        net = build_mlp(8, 4, [], "relu")
        # Une seule Linear, pas d'activation.
        linears = [m for m in net if isinstance(m, nn.Linear)]
        self.assertEqual(len(linears), 1)
        self.assertEqual(linears[0].in_features, 8)
        self.assertEqual(linears[0].out_features, 4)

    def test_no_activation_on_output_layer(self):
        net = build_mlp(8, 4, [128, 128], "relu")
        self.assertIsInstance(net[-1], nn.Linear)
        self.assertEqual(net[-1].out_features, 4)

    def test_layer_count(self):
        net = build_mlp(8, 4, [128, 128], "relu")
        self.assertEqual(len(net), 5)

    def test_intermediate_dimensions(self):
        net = build_mlp(8, 4, [64, 32, 16], "relu")
        linears = [m for m in net if isinstance(m, nn.Linear)]
        dims = [(l.in_features, l.out_features) for l in linears]
        self.assertEqual(dims, [(8, 64), (64, 32), (32, 16), (16, 4)])


class TestMLPActivations(unittest.TestCase):

    def test_relu_activation(self):
        net = build_mlp(8, 4, [16], "relu")
        acts = [m for m in net if isinstance(m, nn.ReLU)]
        self.assertEqual(len(acts), 1)

    def test_tanh_activation(self):
        net = build_mlp(8, 4, [16], "tanh")
        acts = [m for m in net if isinstance(m, nn.Tanh)]
        self.assertEqual(len(acts), 1)

    def test_gelu_activation(self):
        net = build_mlp(8, 4, [16], "gelu")
        acts = [m for m in net if isinstance(m, nn.GELU)]
        self.assertEqual(len(acts), 1)


class TestMLPInit(unittest.TestCase):

    def test_biases_initialized_to_zero(self):
        net = build_mlp(8, 4, [128, 128], "relu")
        for m in net.modules():
            if isinstance(m, nn.Linear):
                self.assertTrue(
                    torch.all(m.bias == 0.0),
                    msg=f"Bias not zero-init in {m}",
                )

    def test_weights_not_all_zero(self):
        net = build_mlp(8, 4, [128, 128], "relu")
        for m in net.modules():
            if isinstance(m, nn.Linear):
                self.assertFalse(
                    torch.all(m.weight == 0.0),
                    msg="Weights all zero — init didn't run",
                )

    def test_weights_have_reasonable_scale(self):
        net = build_mlp(8, 4, [128, 128], "relu")
        for m in net.modules():
            if isinstance(m, nn.Linear):
                std = m.weight.std().item()
                self.assertGreater(std, 0.001)
                self.assertLess(std, 1.0)


class TestMLPGuards(unittest.TestCase):

    def test_invalid_input_dim_raises(self):
        with self.assertRaises(ValueError):
            build_mlp(0, 4, [64], "relu")
        with self.assertRaises(ValueError):
            build_mlp(-1, 4, [64], "relu")

    def test_invalid_output_dim_raises(self):
        with self.assertRaises(ValueError):
            build_mlp(8, 0, [64], "relu")

    def test_invalid_hidden_size_raises(self):
        with self.assertRaises(ValueError):
            build_mlp(8, 4, [128, 0, 64], "relu")
        with self.assertRaises(ValueError):
            build_mlp(8, 4, [-1], "relu")

    def test_unknown_activation_raises(self):
        with self.assertRaises(ValueError) as ctx:
            build_mlp(8, 4, [64], "swish")
        self.assertIn("swish", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
