"""Fast checks of DQN+BC joint objective and retention of supervised gradients."""
import pytest
import torch
import torch.nn.functional as F

from Agents.ftg_bc_regularized_dqn import joint_dqn_bc_loss


def toy_case():
    q = torch.zeros(2, 15, requires_grad=True)
    next_q = torch.ones(2, 15) * 2.0
    actions = torch.tensor([[14], [5]])
    rewards = torch.tensor([[1.0], [2.0]])
    dones = torch.tensor([[0.0], [1.0]])
    expert = torch.zeros(2, 15, requires_grad=True)
    expert_actions = torch.tensor([14, 5])
    return q, actions, rewards, dones, next_q, expert, expert_actions


def test_zero_bc_gives_exact_huber_dqn_objective():
    q, a, r, d, nxt, expert, labels = toy_case()
    total, td, bc = joint_dqn_bc_loss(q, a, r, d, nxt, 0.99,
                                      expert, labels, 0.0)
    expected = F.smooth_l1_loss(torch.zeros(2, 1),
                                torch.tensor([[2.98], [2.0]]))
    torch.testing.assert_close(td, expected)
    torch.testing.assert_close(total, td)
    assert bc.item() == 0.0


def test_joint_loss_backpropagates_into_rl_and_expert_heads():
    q, a, r, d, nxt, expert, labels = toy_case()
    total, td, bc = joint_dqn_bc_loss(q, a, r, d, nxt, 0.99,
                                      expert, labels, 0.1)
    assert total.item() > td.item()
    assert bc.item() > 0
    total.backward()
    assert q.grad is not None and q.grad.abs().sum().item() > 0
    assert expert.grad is not None and expert.grad.abs().sum().item() > 0
    # Learner is pushed toward teacher action 14 for first sample.
    assert expert.grad[0, 14].item() < expert.grad[0, 1].item()


def test_terminal_samples_do_not_bootstrap():
    q, a, r, d, nxt, expert, labels = toy_case()
    total, td, _ = joint_dqn_bc_loss(q, a, r, d, nxt * 100, 0.99,
                                     expert, labels, 0)
    expected = F.smooth_l1_loss(torch.zeros(2, 1),
                                torch.tensor([[199.0], [2.0]]))
    torch.testing.assert_close(td, expected)
    torch.testing.assert_close(total, td)


def test_negative_imitation_weight_rejected():
    with pytest.raises(ValueError, match='nonnegative'):
        joint_dqn_bc_loss(*toy_case()[:5], 0.99, *toy_case()[5:], -0.1)
