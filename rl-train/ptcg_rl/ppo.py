"""MaskablePPO with a faster update: optional fp16 autocast, and no per-minibatch syncs.

SB3's ``MaskablePPO.train()`` calls ``.item()`` five times per minibatch for its
logs, so the GPU drains ~500 times per 65k-sample update and the CPU waits each
time. ``PTCGPPO.train`` keeps those statistics on the GPU and reads them once at
the end; it syncs per minibatch only when ``target_kl`` needs the value.

With ``amp`` the forward and backward passes run under fp16 autocast with a
gradient scaler (weights, optimizer state and the losses stay fp32): on a T4,
whose fp16 tensor cores are ~8x its fp32 throughput, this is most of the update.
bf16 would need no scaler but Turing GPUs (T4) do not support it.

Otherwise the update is SB3's, except that ``train/approx_kl`` and the other
losses are averaged over every epoch rather than SB3's last one.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import explained_variance


class PTCGPPO(MaskablePPO):
    amp: bool = False  # set by train.py; not saved with the model

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "amp", "_scaler"]

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = self.clip_range_vf(self._current_progress_remaining) if self.clip_range_vf is not None else None

        amp = bool(self.amp) and self.device.type == "cuda"
        scaler = getattr(self, "_scaler", None)
        if amp and scaler is None:
            scaler = self._scaler = torch.amp.GradScaler("cuda")

        sums = torch.zeros(5, device=self.device)  # entropy, policy, value, clip fraction, approx KL
        n_batches = 0
        loss = torch.zeros((), device=self.device)
        continue_training = True
        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    values, log_prob, entropy = self.policy.evaluate_actions(
                        rollout_data.observations, actions, action_masks=rollout_data.action_masks)
                values, log_prob = values.float().flatten(), log_prob.float()

                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf)
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                entropy_loss = -torch.mean(-log_prob) if entropy is None else -torch.mean(entropy.float())
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio)
                    clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float())
                    sums += torch.stack([entropy_loss.detach(), policy_loss.detach(), value_loss.detach(),
                                         clip_fraction, approx_kl])
                n_batches += 1

                if self.target_kl is not None and approx_kl.item() > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl.item():.2f}")
                    break

                self.policy.optimizer.zero_grad(set_to_none=True)
                if amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.policy.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    scaler.step(self.policy.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        means = (sums / max(n_batches, 1)).tolist()
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.logger.record("train/entropy_loss", means[0])
        self.logger.record("train/policy_gradient_loss", means[1])
        self.logger.record("train/value_loss", means[2])
        self.logger.record("train/clip_fraction", means[3])
        self.logger.record("train/approx_kl", means[4])
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        if amp:
            self.logger.record("train/amp_scale", scaler.get_scale())
            # The scaler halves its scale on every overflow; one this small means fp16
            # keeps overflowing, so the rest of the run trains in fp32.
            if scaler.get_scale() < 1.0:
                self.amp = False
                print("fp16 update keeps overflowing (grad scale < 1); continuing in fp32", flush=True)
