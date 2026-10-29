import unittest

import torch

from knee_segmentation.model import UNet


class TestUNet(unittest.TestCase):
    def test_output_shape_matches_input_spatial_dims(self):
        model = UNet(in_channels=1, out_channels=1)
        x = torch.randn(2, 1, 128, 160)

        y = model(x)

        self.assertEqual(y.shape, (2, 1, 128, 160))

    def test_backward_pass_runs(self):
        model = UNet(in_channels=1, out_channels=1)
        x = torch.randn(1, 1, 64, 64)
        target = torch.ones(1, 1, 64, 64)

        logits = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        loss.backward()

        has_grad = any(param.grad is not None for param in model.parameters())
        self.assertTrue(has_grad)


if __name__ == "__main__":
    unittest.main()
