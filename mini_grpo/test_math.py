"""Unit tests for the algorithmic pieces that do not require loading Qwen."""

from __future__ import annotations

import unittest

import torch

from mini_grpo.train_minimal_grpo import completion_mask, group_advantages, grpo_loss


class MinimalGrpoMathTests(unittest.TestCase):
    def test_binary_rewards_have_expected_group_advantages(self) -> None:
        advantages = group_advantages(torch.tensor([1.0, 0.0, 0.0, 1.0]))
        torch.testing.assert_close(
            advantages,
            torch.tensor([1.0, -1.0, -1.0, 1.0]),
            atol=3e-4,
            rtol=0,
        )

    def test_equal_rewards_produce_no_policy_signal(self) -> None:
        advantages = group_advantages(torch.tensor([0.0, 0.0, 0.0, 0.0]))
        torch.testing.assert_close(advantages, torch.zeros(4))

    def test_mask_includes_first_eos_only(self) -> None:
        eos = 2
        ids = torch.tensor([[5, eos, eos, eos], [6, 7, 8, 9]])
        expected = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
        torch.testing.assert_close(completion_mask(ids, eos), expected)

    def test_positive_advantage_has_negative_loss_at_initial_policy(self) -> None:
        # When current and old policy match, ratio=1. A positive advantage is
        # therefore rewarded by a negative minimization loss.
        logps = torch.zeros((2, 3), requires_grad=True)
        mask = torch.ones((2, 3))
        loss, _ = grpo_loss(
            policy_logps=logps,
            old_policy_logps=logps.detach(),
            ref_logps=torch.zeros((2, 3)),
            advantages=torch.tensor([1.0, -1.0]),
            mask=mask,
            beta=0.0,
            clip_epsilon=0.2,
        )
        self.assertAlmostEqual(loss.item(), 0.0, places=6)
        loss.backward()
        self.assertLess(logps.grad[0].mean().item(), 0.0)
        self.assertGreater(logps.grad[1].mean().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
