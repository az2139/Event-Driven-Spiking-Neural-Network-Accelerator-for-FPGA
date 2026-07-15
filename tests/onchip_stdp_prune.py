#!/usr/bin/env python3
"""
Pruning-Oriented SNN Trainer
============================

This script is the pruning/training sandbox.  The default route is a
surrogate-BP SNN-style MLP with the same MNIST loading and rate encoding as the
existing faithful route:

    input 784 -> hidden 1000 -> output 10 classes

The output layer reports both class score and class spike count.  The original
faithful R-STDP route is kept as an optional baseline via --trainer stdp.

Usage:
    python tests/onchip_stdp_prune.py --trainer bp
"""

import sys, os, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
try:
    import coregroup_mapping
except ImportError:
    coregroup_mapping = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =====================================================================
# Constants
# =====================================================================
SCALE = 127           # int8 unsigned max (float [0,1] -> int [0,127])
W_MIN_F = 0.01        # SW float clamp min
W_MAX_F = 0.95        # SW float clamp max
W_MIN_I = 1           # round(0.01 * 127)
W_MAX_I = 121         # round(0.95 * 127)


# =====================================================================
# BP Trainer (Rate-encoded SNN-style MLP)
# =====================================================================

class SpikeSTE(torch.autograd.Function):
    """Hard spike in forward, sigmoid surrogate gradient in backward."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(x)
        ctx.scale = scale
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        scale = ctx.scale
        sig = torch.sigmoid(scale * x)
        grad = scale * sig * (1.0 - sig)
        return grad_output * grad, None


class BPPruneNet(nn.Module):
    """Event-driven hidden layer followed by a non-spiking score readout."""

    def __init__(self, n_input=784, hidden_size=1000, n_classes=10,
                 hidden_threshold=1.0, output_threshold=1.0,
                 leak=0.0, surrogate_scale=10.0):
        super().__init__()
        self.n_input = n_input
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.hidden_threshold = hidden_threshold
        self.output_threshold = output_threshold
        self.leak = leak
        self.surrogate_scale = surrogate_scale

        self.fc1 = nn.Linear(n_input, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, n_classes, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        with torch.no_grad():
            self.fc2.weight.abs_()

    def spike(self, x):
        return SpikeSTE.apply(x, self.surrogate_scale)

    def forward(self, spikes, fc1_mask=None, fc2_mask=None,
                random_event_order=None):
        """
        Args:
            spikes: [B, 1, 784] deterministic binary input events.

        Returns:
            score: non-spiking class accumulators driven by hidden spikes.
            active: score readouts which cross output_threshold (diagnostic only).
        """
        B, T, _ = spikes.shape
        if T != 1:
            raise ValueError("event-aware BP forward currently requires T=1")
        if random_event_order is None:
            random_event_order = self.training
        mem1 = torch.zeros(B, self.hidden_size, device=spikes.device)
        fired1 = torch.zeros(B, self.hidden_size, device=spikes.device)
        score = torch.zeros(B, self.n_classes, device=spikes.device)
        fc1_weight = self.fc1.weight if fc1_mask is None else self.fc1.weight * fc1_mask
        fc2_weight = self.fc2.weight if fc2_mask is None else self.fc2.weight * fc2_mask
        frame = spikes[:, 0, :] > 0
        active_count = frame.sum(dim=1)
        max_events = int(active_count.max().item()) if B else 0
        if random_event_order:
            priority = torch.rand(B, self.n_input, device=spikes.device)
        else:
            priority = torch.arange(
                self.n_input, device=spikes.device, dtype=torch.float32
            ).unsqueeze(0).expand(B, -1)
        priority = priority.masked_fill(~frame, float("inf"))
        event_order = priority.argsort(dim=1)

        for event_idx in range(max_events):
            pixel = event_order[:, event_idx]
            valid = (event_idx < active_count).float().unsqueeze(1)
            # Each sample may process a different pixel at this event position.
            delta = fc1_weight[:, pixel].transpose(0, 1) * valid
            mem1 = (mem1 + delta - self.leak * valid).clamp(min=0.0)
            spk1 = self.spike(mem1 - self.hidden_threshold)
            spk1 = spk1 * valid * (1.0 - fired1)
            fired1 = torch.clamp(fired1 + spk1.detach(), max=1.0)
            mem1 = mem1 * (1.0 - spk1.detach())
            score = score + F.linear(spk1, fc2_weight, None)

        active = (score >= self.output_threshold).float()
        return score, active

    def enforce_hardware_constraints(self):
        """Keep the score readout monotonic under event arrival ordering."""
        with torch.no_grad():
            self.fc2.weight.clamp_(min=0.0)


class BPPruneTrainer:
    """BP wrapper using the deterministic input encoding used on the FPGA."""

    def __init__(self, n_input=784, hidden_size=1000, n_classes=10,
                 timesteps=1, hidden_threshold=1.0, output_threshold=1.0,
                 leak=0.0, surrogate_scale=10.0, input_threshold=0.3,
                 device=DEVICE):
        self.n_input = n_input
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.timesteps = timesteps
        self.input_threshold = float(input_threshold)
        self.device = device
        self.net = BPPruneNet(
            n_input=n_input, hidden_size=hidden_size, n_classes=n_classes,
            hidden_threshold=hidden_threshold, output_threshold=output_threshold,
            leak=leak, surrogate_scale=surrogate_scale).to(device)
        self.fc1_mask = None
        self.fc2_mask = None
        self.mapping_history = []
        self.best_epoch = -1
        self.matching_mapping = ''

    def rate_encode(self, images):
        B = images.shape[0]
        frame = (images > self.input_threshold).float()
        return frame.unsqueeze(1).expand(B, self.timesteps, self.n_input)

    def forward(self, spikes, use_mask=True, random_event_order=None):
        fc1_mask = self.fc1_mask if use_mask else None
        fc2_mask = self.fc2_mask if use_mask else None
        return self.net(
            spikes, fc1_mask=fc1_mask, fc2_mask=fc2_mask,
            random_event_order=random_event_order)

    @torch.no_grad()
    def test_batch(self, test_imgs, test_lbls, batch_size=256):
        self.net.eval()
        N = len(test_imgs)
        flat = test_imgs.reshape(N, -1).to(self.device).float()
        lbls = test_lbls.to(self.device)

        score_correct = 0
        count_correct = 0
        per_class_score_c = torch.zeros(self.n_classes, device=self.device)
        per_class_count_c = torch.zeros(self.n_classes, device=self.device)
        per_class_t = torch.zeros(self.n_classes, device=self.device)

        rng_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        torch.manual_seed(12345)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(12345)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            spikes = self.rate_encode(flat[start:end])
            score, count = self.forward(spikes, use_mask=True)
            score_pred = score.argmax(dim=1)
            count_pred = self._count_pred(count, score)
            y = lbls[start:end]

            score_correct += (score_pred == y).sum().item()
            count_correct += (count_pred == y).sum().item()
            for c in range(self.n_classes):
                mask = y == c
                per_class_t[c] += mask.sum()
                per_class_score_c[c] += ((score_pred == c) & mask).sum()
                per_class_count_c[c] += ((count_pred == c) & mask).sum()

        torch.random.set_rng_state(rng_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)

        return {
            'score_acc': score_correct / N * 100,
            'count_acc': count_correct / N * 100,
            'per_class_score_c': per_class_score_c,
            'per_class_count_c': per_class_count_c,
            'per_class_t': per_class_t,
        }

    @torch.no_grad()
    def test_int8_reference(self, test_imgs, test_lbls, batch_size=256,
                            input_weight=255):
        """Deterministic event-order hidden layer with INT8 score readout."""
        if self.timesteps != 1:
            raise ValueError("INT8 deployment reference requires timesteps=1")
        if self.fc1_mask is None or self.fc2_mask is None:
            raise ValueError("INT8 deployment reference requires pruning masks")

        threshold = float(self.net.hidden_threshold)
        scale = float(input_weight) / threshold
        q1 = torch.round(self.net.fc1.weight * scale).clamp(-255, 255)
        q2 = torch.round(self.net.fc2.weight * scale).clamp(-255, 255)
        q1 = q1 * self.fc1_mask
        q2 = q2 * self.fc2_mask
        hw_threshold = int(round(threshold * scale))

        flat = test_imgs.reshape(len(test_imgs), -1).to(self.device).float()
        labels = test_lbls.to(self.device)
        correct = 0
        hidden_spikes = 0
        active_readouts = 0
        for start in range(0, len(flat), batch_size):
            end = min(start + batch_size, len(flat))
            frame = flat[start:end] > self.input_threshold
            B = len(frame)
            hidden_mem = torch.zeros(B, self.hidden_size, device=self.device)
            hidden = torch.zeros(B, self.hidden_size, dtype=torch.bool,
                                 device=self.device)
            active_count = frame.sum(dim=1)
            max_events = int(active_count.max().item()) if B else 0
            priority = torch.arange(
                self.n_input, device=self.device, dtype=torch.float32
            ).unsqueeze(0).expand(B, -1).masked_fill(~frame, float("inf"))
            event_order = priority.argsort(dim=1)
            for event_idx in range(max_events):
                pixel = event_order[:, event_idx]
                valid = (event_idx < active_count).unsqueeze(1)
                delta = q1[:, pixel].transpose(0, 1) * valid
                hidden_mem = (hidden_mem + delta).clamp(min=0, max=65535)
                new_spike = valid & ~hidden & (hidden_mem >= hw_threshold)
                hidden |= new_spike
                hidden_mem = hidden_mem.masked_fill(new_spike, 0)
            output_score = F.linear(hidden.float(), q2, None)
            pred = output_score.argmax(dim=1)
            correct += (pred == labels[start:end]).sum().item()
            hidden_spikes += hidden.sum().item()
            active_readouts += (output_score >= hw_threshold).sum().item()

        total = len(flat)
        return {
            'accuracy': correct / max(total, 1) * 100.0,
            'hidden_spikes_per_sample': hidden_spikes / max(total, 1),
            'active_readouts_per_sample': active_readouts / max(total, 1),
            'samples': int(total),
            'weight_scale': scale,
            'hardware_threshold': hw_threshold,
        }

    @staticmethod
    def _count_pred(count, score):
        # Score is the tie-breaker when multiple classes have equal counts.
        max_count = count.max(dim=1, keepdim=True).values
        tied_score = score.masked_fill(count != max_count, -1e30)
        return tied_score.argmax(dim=1)

    def save_model(self, path):
        net = self.net
        np.savez(path,
                 fc1_weight=net.fc1.weight.detach().cpu().numpy(),
                 fc2_weight=net.fc2.weight.detach().cpu().numpy(),
                 fc1_mask=(np.asarray([], dtype=np.float32)
                           if self.fc1_mask is None else self.fc1_mask.detach().cpu().numpy()),
                 fc2_mask=(np.asarray([], dtype=np.float32)
                           if self.fc2_mask is None else self.fc2_mask.detach().cpu().numpy()),
                 mapping_history=np.asarray(self.mapping_history, dtype=object),
                 best_epoch=np.array(self.best_epoch, dtype=np.int32),
                 matching_mapping=np.array(self.matching_mapping),
                 hidden_threshold=net.hidden_threshold,
                 output_threshold=net.output_threshold,
                 leak=net.leak,
                 timesteps=self.timesteps,
                 input_encoding='deterministic_threshold',
                 input_threshold=self.input_threshold,
                 hidden_execution='event_serial_fire_once_reset_zero',
                 training_event_order='random_per_sample',
                 evaluation_event_order='ascending_input_id',
                 output_readout='nonnegative_weight_score_argmax',
                 n_input=self.n_input,
                 hidden_size=self.hidden_size,
                 n_classes=self.n_classes,
                 training='bp_surrogate_event_hidden_score_readout')
        print(f"  Saved -> {path}")

    def set_prune_masks(self, fc1_mask, fc2_mask):
        self.fc1_mask = torch.as_tensor(fc1_mask, dtype=torch.float32, device=self.device)
        self.fc2_mask = torch.as_tensor(fc2_mask, dtype=torch.float32, device=self.device)

    def apply_prune_masks(self):
        # New mainline keeps dense weights intact.  Masks are applied only in
        # forward(), so inactive edges can later regrow from their stored weight.
        return

    def enforce_hardware_constraints(self):
        self.net.enforce_hardware_constraints()

    def mask_gradients(self):
        if self.fc1_mask is not None and self.net.fc1.weight.grad is not None:
            self.net.fc1.weight.grad.mul_(self.fc1_mask)
        if self.fc2_mask is not None and self.net.fc2.weight.grad is not None:
            self.net.fc2.weight.grad.mul_(self.fc2_mask)

    def snapshot_inactive_weights(self):
        if self.fc1_mask is None or self.fc2_mask is None:
            return None
        with torch.no_grad():
            return (
                self.net.fc1.weight.detach().clone(),
                self.net.fc2.weight.detach().clone(),
            )

    def restore_inactive_weights(self, snapshot):
        if snapshot is None or self.fc1_mask is None or self.fc2_mask is None:
            return
        old_fc1, old_fc2 = snapshot
        with torch.no_grad():
            self.net.fc1.weight.copy_(
                self.net.fc1.weight * self.fc1_mask + old_fc1 * (1.0 - self.fc1_mask)
            )
            self.net.fc2.weight.copy_(
                self.net.fc2.weight * self.fc2_mask + old_fc2 * (1.0 - self.fc2_mask)
            )


# =====================================================================
# Faithful On-Chip Trainer (Float weights + Int8 forward)
# =====================================================================

class FaithfulOnChipTrainer:
    """
    Identical to DenseSTDP10Class but forward pass uses int8 weights.

    Training: float weights, float STDP (same formulas as SW).
    Forward: weights quantized to int8 for inference (matching HLS).
    Result: same weights as SW, same accuracy.

    This represents: "Train in software, deploy on FPGA int8 inference."
    """

    def __init__(self, n_input=784, n_classes=10, features_per_class=10,
                 threshold=None, leak=0.0, timesteps=1, input_threshold=0.3,
                 lr_plus=0.005, lr_minus=0.003,
                 device=DEVICE):
        self.n_input = n_input
        self.n_classes = n_classes
        self.features_per_class = features_per_class
        self.n_output = n_classes * features_per_class
        self.device = device
        self.timesteps = timesteps
        self.input_threshold = float(input_threshold)
        self.lr_plus = lr_plus
        self.lr_minus = lr_minus
        self.leak = leak

        # Decision map
        self.decision_map = torch.arange(self.n_output, device=device) // features_per_class
        self.class_starts = [c * features_per_class for c in range(n_classes)]
        self.class_ends = [(c + 1) * features_per_class for c in range(n_classes)]

        # Weights: float32, same as SW
        self.weights = torch.empty(self.n_output, n_input, device=device)
        self.weights.uniform_(0.3, 0.7)

        # Threshold
        if threshold is None:
            self.base_threshold = 80.0
        else:
            self.base_threshold = threshold
        self.thresholds = torch.full((self.n_output,), self.base_threshold,
                                      device=device, dtype=torch.float32)
        self.adapt_rate = 2.0
        self.target_fire_rate = 0.05

        self.fire_counts = torch.zeros(self.n_output, device=device)
        self.sample_count = 0
        self._class_correct = torch.zeros(n_classes, device=device)
        self._class_total = torch.zeros(n_classes, device=device)

        # Training vs inference mode:
        # - training=True:  forward uses float weights (identical to SW)
        # - training=False: forward uses int8-quantized weights (FPGA behavior)
        self.training = True

    def train(self):
        """Set to training mode (float forward, identical to SW)."""
        self.training = True

    def eval(self):
        """Set to eval mode (int8 forward, matching FPGA inference)."""
        self.training = False

    def _get_int8_weights(self):
        """Quantize float weights to int8 scale for forward pass."""
        return (self.weights * SCALE).round().clamp(W_MIN_I, W_MAX_I)

    # -- Prototype init (identical to SW) ----------------------------------
    def init_prototypes(self, train_imgs, train_lbls):
        flat = train_imgs.reshape(len(train_imgs), -1).to(self.device)
        lbls = train_lbls.to(self.device)

        for c in range(self.n_classes):
            mask = lbls == c
            if mask.any():
                class_mean = flat[mask].mean(dim=0)
                s, e = self.class_starts[c], self.class_ends[c]
                for f in range(self.features_per_class):
                    noise = torch.randn(self.n_input, device=self.device) * 0.08
                    self.weights[s + f] = (class_mean + noise).clamp(0.05, 0.95)

        self._normalize_weights()
        w_int = self._get_int8_weights()
        print(f"  Prototype init: W=[{self.weights.min():.3f},{self.weights.max():.3f}] "
              f"(int8: [{int(w_int.min())},{int(w_int.max())}])")
        self._calibrate_threshold(flat[:200])

    def _normalize_weights(self):
        """Same as SW._normalize_weights."""
        norms = self.weights.norm(dim=1, keepdim=True).clamp(min=1e-6)
        target_norm = (self.n_input * 0.02) ** 0.5
        self.weights *= target_norm / norms
        self.weights.clamp_(W_MIN_F, W_MAX_F)

    def _calibrate_threshold(self, sample_imgs):
        """Same as SW._calibrate_threshold but using int8 weights for forward."""
        B = sample_imgs.shape[0]
        spikes = self.rate_encode(sample_imgs)
        inp_mean = spikes.mean(dim=1)
        # Use int8 weights for realistic threshold calibration
        w_int = self._get_int8_weights()
        pot_per_step = (inp_mean @ w_int.T).mean(dim=0) / SCALE  # back to float scale
        target_t = 6
        new_thr = pot_per_step.mean().item() * target_t
        new_thr = max(new_thr, 5.0)
        self.base_threshold = new_thr
        self.thresholds.fill_(new_thr)
        print(f"  Calibrated threshold: {new_thr:.1f} (int8 equiv: {new_thr * SCALE:.0f})")

    # -- Deterministic encoding used by the board input path ----------------
    def rate_encode(self, images):
        B = images.shape[0]
        T = self.timesteps
        frame = (images > self.input_threshold).float()
        return frame.unsqueeze(1).expand(B, T, self.n_input)

    # -- Forward pass (int8-quantized weights) -----------------------------
    @torch.no_grad()
    def forward_batch(self, spikes):
        """
        Forward with WTA.

        Training mode: uses float weights (identical to SW DenseSTDP10Class).
        Eval mode:     uses int8-quantized weights (matching HLS/FPGA inference).

        This ensures training produces IDENTICAL weights to SW,
        while eval shows the actual FPGA deployment accuracy.
        """
        B, T, _ = spikes.shape
        N = self.n_output
        if self.training:
            w = self.weights  # float (identical to SW)
        else:
            w = self._get_int8_weights() / SCALE  # int8-resolution float
        potentials = torch.zeros(B, N, device=self.device)
        fired = torch.zeros(B, N, dtype=torch.bool, device=self.device)
        fire_times = torch.full((B, N), T + 1, dtype=torch.float32, device=self.device)
        thresholds = self.thresholds.unsqueeze(0)

        for t in range(T):
            inp = spikes[:, t, :]
            potentials += inp @ w.T
            potentials -= self.leak
            potentials.clamp_(min=0)

            above = (potentials >= thresholds) & ~fired
            if above.any():
                pot_cand = potentials.clone()
                pot_cand[~above] = -1.0
                pot_cand += torch.rand_like(pot_cand) * 0.01
                winners = pot_cand.argmax(dim=1)
                has_winner = above.any(dim=1)

                if has_winner.any():
                    batch_idx = torch.arange(B, device=self.device)[has_winner]
                    neuron_idx = winners[has_winner]
                    fired[batch_idx, neuron_idx] = True
                    fire_times[batch_idx, neuron_idx] = float(t) + torch.rand(
                        len(batch_idx), device=self.device) * 0.1
                    potentials[has_winner] = 0.0

        preds = torch.empty(B, dtype=torch.long, device=self.device)
        any_fired = fired.any(dim=1)
        if any_fired.any():
            ft_masked = fire_times.clone()
            ft_masked[~fired] = T + 2.0
            winners = ft_masked.argmin(dim=1)
            preds[any_fired] = self.decision_map[winners[any_fired]]
        if (~any_fired).any():
            pot_nf = potentials + torch.rand_like(potentials) * 0.01
            winners_nf = pot_nf.argmax(dim=1)
            preds[~any_fired] = self.decision_map[winners_nf[~any_fired]]

        return preds, potentials, fired, fire_times

    # -- R-STDP (identical to SW, float arithmetic) ------------------------
    @torch.no_grad()
    def train_rstdp_batch(self, spikes, targets, predictions, fired, fire_times):
        """Identical to SW DenseSTDP10Class.train_rstdp_batch."""
        pre_activity = spikes.sum(dim=1)
        pre_active = (pre_activity > 0).float()
        correct = predictions == targets

        lr_p = self.lr_plus
        lr_m = self.lr_minus
        B = spikes.shape[0]

        for i in range(B):
            active = pre_active[i]
            pred_c = predictions[i].item()
            target_c = targets[i].item()

            if correct[i]:
                s, e = self.class_starts[pred_c], self.class_ends[pred_c]
                if fired[i, s:e].any():
                    winner_local = fire_times[i, s:e].argmin().item()
                else:
                    sims = (self.weights[s:e] * active.unsqueeze(0)).sum(dim=1)
                    winner_local = sims.argmax().item()

                w = self.weights[s + winner_local]
                w += lr_p * active * (1.0 - w)

            else:
                s_p, e_p = self.class_starts[pred_c], self.class_ends[pred_c]
                if fired[i, s_p:e_p].any():
                    wp_local = fire_times[i, s_p:e_p].argmin().item()
                else:
                    sims = (self.weights[s_p:e_p] * active.unsqueeze(0)).sum(dim=1)
                    wp_local = sims.argmax().item()
                w = self.weights[s_p + wp_local]
                w -= lr_m * active * w

                s_t, e_t = self.class_starts[target_c], self.class_ends[target_c]
                sims = (self.weights[s_t:e_t] * active.unsqueeze(0)).sum(dim=1)
                wt_local = sims.argmax().item()
                w = self.weights[s_t + wt_local]
                w += lr_p * active * (1.0 - w)

        self.weights.clamp_(W_MIN_F, W_MAX_F)

        self.fire_counts += fired.float().sum(dim=0)
        self.sample_count += B

        for c in range(self.n_classes):
            mask_c = targets == c
            if mask_c.any():
                self._class_total[c] += mask_c.sum()
                self._class_correct[c] += (correct & mask_c).sum()

    def adapt_thresholds(self):
        """Identical to SW."""
        if self.sample_count == 0:
            return
        fire_rates = self.fire_counts / self.sample_count
        delta = self.adapt_rate * (fire_rates - self.target_fire_rate)
        self.thresholds += delta
        self.thresholds.clamp_(min=self.base_threshold * 0.3,
                                max=self.base_threshold * 3.0)
        self.fire_counts.zero_()
        self.sample_count = 0

    def normalize_weights(self):
        """Same as SW gentle norm."""
        with torch.no_grad():
            self.weights.clamp_(W_MIN_F, W_MAX_F)
            norms = self.weights.norm(dim=1, keepdim=True).clamp(min=1e-6)
            median_norm = norms.median()
            scale = (median_norm / norms).clamp(0.9, 1.1)
            self.weights *= scale
            self.weights.clamp_(W_MIN_F, W_MAX_F)

    # -- Test (same as SW) -------------------------------------------------
    @torch.no_grad()
    def test_batch(self, test_imgs, test_lbls, batch_size=256):
        """Test accuracy using int8 forward (FPGA-realistic inference)."""
        was_training = self.training
        self.eval()  # int8 forward for FPGA-realistic accuracy

        N = len(test_imgs)
        flat = test_imgs.reshape(N, -1).to(self.device).float()
        lbls = test_lbls.to(self.device)

        rng_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        torch.manual_seed(12345)
        if torch.cuda.is_available(): torch.cuda.manual_seed(12345)

        per_class_c = torch.zeros(self.n_classes, device=self.device)
        per_class_t = torch.zeros(self.n_classes, device=self.device)

        all_preds = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            spikes = self.rate_encode(flat[start:end])
            preds, _, _, _ = self.forward_batch(spikes)
            all_preds.append(preds)
            for c in range(self.n_classes):
                mask = lbls[start:end] == c
                per_class_t[c] += mask.sum()
                per_class_c[c] += ((preds == c) & mask).sum()

        all_preds = torch.cat(all_preds)
        correct = (all_preds == lbls).sum().item()
        acc = correct / N * 100

        torch.random.set_rng_state(rng_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)

        if was_training:
            self.train()  # restore training mode

        return acc, per_class_c, per_class_t

    def quantize_weights(self, scale=127.0):
        """Quantize to int8 for FPGA deployment."""
        w_np = self.weights.cpu().numpy()
        q = np.round(w_np * scale).astype(np.int8)
        return q, scale

    def get_weight_stats(self):
        w = self.weights
        w_int = self._get_int8_weights()
        return {
            'min_f': float(w.min()), 'max_f': float(w.max()),
            'mean_f': float(w.mean()), 'std_f': float(w.std()),
            'min_i': int(w_int.min()), 'max_i': int(w_int.max()),
            'mean_i': float(w_int.mean()), 'std_i': float(w_int.std()),
        }

    def save_model(self, path):
        """Save model with both float and int8 weights."""
        q, sc = self.quantize_weights()
        np.savez(path,
                 weights=self.weights.cpu().numpy(),
                 weights_int8=q,
                 thresholds=self.thresholds.cpu().numpy(),
                 decision_map=self.decision_map.cpu().numpy(),
                 threshold=self.base_threshold, leak=self.leak,
                 timesteps=self.timesteps, n_input=self.n_input,
                 input_encoding='deterministic_threshold',
                 input_threshold=self.input_threshold,
                 n_classes=self.n_classes,
                 features_per_class=self.features_per_class,
                 scale=SCALE)
        print(f"  Saved -> {path}")



# =====================================================================
# Data Loading
# =====================================================================

def load_mnist(data_dir='./data', max_train=None, max_test=None):
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    def to_tensors(ds, max_n=None):
        imgs, lbls = [], []
        for img, lbl in ds:
            imgs.append(img.squeeze(0))
            lbls.append(lbl)
            if max_n and len(imgs) >= max_n:
                break
        return torch.stack(imgs), torch.tensor(lbls, dtype=torch.long)

    return to_tensors(train_ds, max_train), to_tensors(test_ds, max_test)


# =====================================================================
# Training Loop
# =====================================================================

def compute_input_rates_for_mapping(train_imgs, n_input):
    imgs = train_imgs.reshape(len(train_imgs), -1).detach().cpu().numpy().astype(np.float32)
    return np.maximum(imgs.mean(axis=0), 1e-6)


def update_mapping_and_prune(trainer, input_rates, epoch, args, snapshot_prefix=None):
    if coregroup_mapping is None:
        raise RuntimeError("coregroup_mapping.py is not importable")

    fc1 = trainer.net.fc1.weight.detach().cpu().numpy()
    fc2 = trainer.net.fc2.weight.detach().cpu().numpy()
    model = coregroup_mapping.model_from_arrays(fc1, fc2)

    if trainer.fc1_mask is None or trainer.fc2_mask is None:
        init_masks = coregroup_mapping.build_initial_topk_masks(
            model=model,
            input_rates=input_rates,
            input_hidden_topk=args.map_input_hidden_topk,
            hidden_output_topk=args.map_hidden_output_topk,
            min_importance=args.map_min_abs_weight,
            importance_mode=args.importance_mode,
        )
        trainer.set_prune_masks(init_masks["fc1_mask"], init_masks["fc2_mask"])

    old_fc1_mask = trainer.fc1_mask.detach().cpu().numpy().astype(np.float32)
    old_fc2_mask = trainer.fc2_mask.detach().cpu().numpy().astype(np.float32)
    mapping = coregroup_mapping.run_mapping(
        model=model,
        input_rates=input_rates,
        num_groups=args.map_num_groups,
        neurons_per_group=args.map_neurons_per_group,
        input_hidden_topk=args.map_input_hidden_topk,
        hidden_output_topk=args.map_hidden_output_topk,
        min_abs_weight=args.map_min_abs_weight,
        fc1_mask=old_fc1_mask,
        fc2_mask=old_fc2_mask,
        load_cap=args.map_load_cap,
        local_edge_cap=args.map_local_edge_cap,
        lambda_load=args.map_lambda_load,
        lambda_cap=args.map_lambda_cap,
        lambda_balance=args.map_lambda_balance,
        max_iter=args.map_max_iter,
    )
    masks = coregroup_mapping.build_cost_aware_masks(
        model=model,
        mapping=mapping,
        input_rates=input_rates,
        input_hidden_topk=args.map_input_hidden_topk,
        hidden_output_topk=args.map_hidden_output_topk,
        cross_rate_coeff=args.runtime_cross_rate_coeff,
        load_balance_coeff=args.runtime_load_balance_coeff,
        load_reward_cap=args.runtime_load_reward_cap,
        load_penalty_cap=args.runtime_load_penalty_cap,
        min_abs_weight=args.map_min_abs_weight,
        runtime_alpha=args.runtime_alpha,
        importance_mode=args.importance_mode,
    )
    trainer.set_prune_masks(masks["fc1_mask"], masks["fc2_mask"])

    new_fc1_mask = masks["fc1_mask"].astype(np.float32)
    new_fc2_mask = masks["fc2_mask"].astype(np.float32)
    fc1_pruned = int(np.logical_and(old_fc1_mask > 0, new_fc1_mask == 0).sum())
    fc1_regrown = int(np.logical_and(old_fc1_mask == 0, new_fc1_mask > 0).sum())
    fc2_pruned = int(np.logical_and(old_fc2_mask > 0, new_fc2_mask == 0).sum())
    fc2_regrown = int(np.logical_and(old_fc2_mask == 0, new_fc2_mask > 0).sum())

    summary = {
        "epoch": int(epoch),
        "fc1_density": float(masks["fc1_density"]),
        "fc2_density": float(masks["fc2_density"]),
        "fc1_pruned": fc1_pruned,
        "fc1_regrown": fc1_regrown,
        "fc2_pruned": fc2_pruned,
        "fc2_regrown": fc2_regrown,
        "mean_group_event_load": float(masks["mean_group_event_load"]),
        "group_load_delta_norm_min": float(np.min(masks["group_load_delta_norm"])),
        "group_load_delta_norm_max": float(np.max(masks["group_load_delta_norm"])),
        **mapping["summary"],
    }
    trainer.mapping_history.append(summary)

    if snapshot_prefix:
        os.makedirs(os.path.dirname(snapshot_prefix) or ".", exist_ok=True)
        path = f"{snapshot_prefix}_epoch{epoch:03d}.npz"
        ns = argparse.Namespace(
            model="in_memory_training_weights",
            output=path,
            num_groups=args.map_num_groups,
            neurons_per_group=args.map_neurons_per_group,
            input_hidden_topk=args.map_input_hidden_topk,
            hidden_output_topk=args.map_hidden_output_topk,
            load_cap=mapping["load_cap"],
            local_edge_cap=mapping["local_edge_cap"],
        )
        coregroup_mapping.write_outputs(
            path,
            mapping["graph"],
            mapping["neuron_group"],
            mapping["neuron_local"],
            mapping["neuron_global_id"],
            mapping["summary"],
            ns,
        )

    print(f"  [Map/Prune] epoch={epoch} "
          f"fc1_density={summary['fc1_density']:.4f} "
          f"fc2_density={summary['fc2_density']:.4f} "
          f"regrow/prune fc1={fc1_regrown}/{fc1_pruned} "
          f"fc2={fc2_regrown}/{fc2_pruned} "
          f"local_event={summary['local_event_ratio']*100:.2f}% "
          f"local_edges={summary['local_edge_ratio']*100:.2f}% "
          f"max_group={summary['max_group_count']}/{args.map_neurons_per_group}")
    return summary


def write_matching_mapping(trainer, input_rates, args, path):
    """Map the restored checkpoint without changing its deployment masks."""
    if trainer.fc1_mask is None or trainer.fc2_mask is None:
        raise RuntimeError("cannot write a deployment mapping without both pruning masks")

    model = coregroup_mapping.model_from_arrays(
        trainer.net.fc1.weight.detach().cpu().numpy(),
        trainer.net.fc2.weight.detach().cpu().numpy(),
    )
    mapping = coregroup_mapping.run_mapping(
        model=model,
        input_rates=input_rates,
        num_groups=args.map_num_groups,
        neurons_per_group=args.map_neurons_per_group,
        input_hidden_topk=args.map_input_hidden_topk,
        hidden_output_topk=args.map_hidden_output_topk,
        min_abs_weight=args.map_min_abs_weight,
        fc1_mask=trainer.fc1_mask.detach().cpu().numpy(),
        fc2_mask=trainer.fc2_mask.detach().cpu().numpy(),
        load_cap=args.map_load_cap,
        local_edge_cap=args.map_local_edge_cap,
        lambda_load=args.map_lambda_load,
        lambda_cap=args.map_lambda_cap,
        lambda_balance=args.map_lambda_balance,
        max_iter=args.map_max_iter,
    )
    ns = argparse.Namespace(
        model="matching_best_training_checkpoint",
        output=path,
        num_groups=args.map_num_groups,
        neurons_per_group=args.map_neurons_per_group,
        input_hidden_topk=args.map_input_hidden_topk,
        hidden_output_topk=args.map_hidden_output_topk,
        load_cap=mapping["load_cap"],
        local_edge_cap=mapping["local_edge_cap"],
    )
    coregroup_mapping.write_outputs(
        path, mapping["graph"], mapping["neuron_group"],
        mapping["neuron_local"], mapping["neuron_global_id"],
        mapping["summary"], ns,
    )
    return mapping["summary"]


def train_bp_model(trainer, train_imgs, train_lbls, epochs=50,
                   batch_size=128, save_path=None,
                   val_imgs=None, val_lbls=None, patience=10,
                   lr=1e-3, weight_decay=1e-4,
                   alternating_args=None):
    """Surrogate-BP training for event-hidden, non-spiking score readout."""
    N = len(train_imgs)
    flat = train_imgs.reshape(N, -1).to(trainer.device).float()
    lbls = train_lbls.to(trainer.device)

    has_val = val_imgs is not None and val_lbls is not None
    optimizer = torch.optim.AdamW(
        trainer.net.parameters(), lr=lr, weight_decay=weight_decay)

    best_metric = float("-inf")
    best_state = None
    best_fc1_mask = None
    best_fc2_mask = None
    best_mapping_history = None
    best_epoch = -1
    no_improve = 0
    input_rates = None
    snapshot_prefix = None
    if alternating_args is not None and alternating_args.enable_alt_mapping_prune:
        input_rates = compute_input_rates_for_mapping(train_imgs, trainer.n_input)
        snapshot_prefix = alternating_args.mapping_snapshot_prefix

    for epoch in range(epochs):
        if (alternating_args is not None and
                alternating_args.enable_alt_mapping_prune and
                epoch >= int(alternating_args.map_warmup_epochs) and
                (epoch - int(alternating_args.map_warmup_epochs)) %
                max(1, int(alternating_args.map_interval_epochs)) == 0):
            update_mapping_and_prune(
                trainer, input_rates, epoch=epoch, args=alternating_args,
                snapshot_prefix=snapshot_prefix)

        trainer.net.train()
        perm = torch.randperm(N, device=trainer.device)
        loss_sum = 0.0
        score_correct = 0
        count_correct = 0
        total = 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            idx = perm[start:end]
            imgs_b = flat[idx]
            lbls_b = lbls[idx]

            spikes = trainer.rate_encode(imgs_b)
            score, count = trainer.forward(spikes, use_mask=True)
            loss = F.cross_entropy(score / max(trainer.timesteps, 1), lbls_b)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            trainer.mask_gradients()
            torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 1.0)
            inactive_snapshot = trainer.snapshot_inactive_weights()
            optimizer.step()
            trainer.restore_inactive_weights(inactive_snapshot)
            trainer.enforce_hardware_constraints()

            with torch.no_grad():
                score_pred = score.argmax(dim=1)
                count_pred = trainer._count_pred(count, score)
                score_correct += (score_pred == lbls_b).sum().item()
                count_correct += (count_pred == lbls_b).sum().item()
                loss_sum += loss.item() * len(lbls_b)
                total += len(lbls_b)

        train_score_acc = score_correct / total * 100
        train_count_acc = count_correct / total * 100
        train_loss = loss_sum / total

        if has_val and (epoch % 2 == 0 or epoch == epochs - 1):
            val_res = trainer.test_batch(val_imgs, val_lbls, batch_size=256)
            val_score_acc = val_res['score_acc']
            val_count_acc = val_res['count_acc']
        elif not has_val:
            val_score_acc = train_score_acc
            val_count_acc = train_count_acc
        else:
            val_score_acc = best_metric
            val_count_acc = 0.0

        if epoch % 2 == 0 or epoch == epochs - 1:
            val_str = (f"  ValScore={val_score_acc:5.1f}%"
                       f"  ValReadout={val_count_acc:5.1f}%") if has_val else ""
            print(f"  Epoch {epoch:3d}: Loss={train_loss:.4f}  "
                  f"TrainScore={train_score_acc:5.1f}%  "
                  f"TrainReadout={train_count_acc:5.1f}%{val_str}")

        metric = val_score_acc if has_val else train_score_acc
        deployable = (input_rates is None or
                      (trainer.fc1_mask is not None and trainer.fc2_mask is not None))
        if deployable and metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in trainer.net.state_dict().items()}
            best_fc1_mask = None if trainer.fc1_mask is None else trainer.fc1_mask.detach().cpu().clone()
            best_fc2_mask = None if trainer.fc2_mask is None else trainer.fc2_mask.detach().cpu().clone()
            best_mapping_history = list(trainer.mapping_history)
            no_improve = 0
        elif deployable:
            no_improve += 1

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        trainer.net.load_state_dict(best_state)
        trainer.net.to(trainer.device)
        trainer.fc1_mask = None if best_fc1_mask is None else best_fc1_mask.to(trainer.device)
        trainer.fc2_mask = None if best_fc2_mask is None else best_fc2_mask.to(trainer.device)
        trainer.mapping_history = ([] if best_mapping_history is None
                                   else best_mapping_history)
        trainer.apply_prune_masks()
        trainer.best_epoch = best_epoch
        if input_rates is not None:
            if not snapshot_prefix:
                base = os.path.splitext(save_path or "bp_coregroup_mapping")[0]
                snapshot_prefix = base + "_mapping"
            mapping_path = f"{snapshot_prefix}_best.npz"
            os.makedirs(os.path.dirname(mapping_path) or ".", exist_ok=True)
            summary = write_matching_mapping(
                trainer, input_rates, alternating_args, mapping_path)
            trainer.matching_mapping = os.path.abspath(mapping_path)
            print(f"  Best deployable checkpoint: epoch={best_epoch} "
                  f"metric={best_metric:.2f}%")
            print(f"  Matching mapping -> {mapping_path} "
                  f"(local_event={summary['local_event_ratio']*100:.2f}%)")
        if save_path:
            trainer.save_model(save_path)
    elif input_rates is not None:
        raise RuntimeError(
            "training ended before a deployable mapped checkpoint was produced; "
            "reduce --map-warmup-epochs or increase --epochs"
        )
    return trainer


def train_model(trainer, train_imgs, train_lbls, epochs=200,
                batch_size=128, save_path=None,
                val_imgs=None, val_lbls=None, patience=20):
    """Train loop - same structure as SW train_network."""
    N = len(train_imgs)
    flat = train_imgs.reshape(N, -1).to(trainer.device).float()
    lbls = train_lbls.to(trainer.device)

    has_val = val_imgs is not None and val_lbls is not None
    if has_val:
        val_flat = val_imgs.reshape(len(val_imgs), -1).to(trainer.device).float()
        val_lbls_d = val_lbls.to(trainer.device)

    best_acc = 0.0
    best_val_acc = 0.0
    best_weights = None
    best_thresholds = None
    no_improve = 0

    for epoch in range(epochs):
        trainer._class_correct.zero_()
        trainer._class_total.zero_()
        perm = torch.randperm(N, device=trainer.device)
        correct = 0
        total = 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            idx = perm[start:end]
            imgs_b = flat[idx]
            lbls_b = lbls[idx]

            spikes = trainer.rate_encode(imgs_b)
            preds, pots, fired, ftimes = trainer.forward_batch(spikes)
            trainer.train_rstdp_batch(spikes, lbls_b, preds, fired, ftimes)

            correct += (preds == lbls_b).sum().item()
            total += len(lbls_b)

        train_acc = correct / total * 100
        trainer.adapt_thresholds()

        if epoch % 5 == 0:
            trainer.normalize_weights()

        if has_val and (epoch % 2 == 0 or epoch == epochs - 1):
            trainer.eval()  # int8 forward for realistic val accuracy
            val_correct = 0
            for vs in range(0, len(val_flat), 256):
                ve = min(vs + 256, len(val_flat))
                vspikes = trainer.rate_encode(val_flat[vs:ve])
                vpreds, _, _, _ = trainer.forward_batch(vspikes)
                val_correct += (vpreds == val_lbls_d[vs:ve]).sum().item()
            val_acc = val_correct / len(val_flat) * 100
            trainer.train()  # restore training mode
        elif not has_val:
            val_acc = train_acc
        else:
            val_acc = best_val_acc

        if epoch % 5 == 0 or epoch == epochs - 1:
            ws = trainer.get_weight_stats()
            ca = trainer._class_correct / trainer._class_total.clamp(min=1) * 100
            thr_min = trainer.thresholds.min().item()
            thr_max = trainer.thresholds.max().item()
            val_str = f"  Val={val_acc:5.1f}%" if has_val else ""
            print(f"  Epoch {epoch:3d}: Train={train_acc:5.1f}%{val_str}  "
                  f"W_mean={ws['mean_f']:.3f}  Thr=[{thr_min:.0f},{thr_max:.0f}]  "
                  f"Class=[{ca.min():.0f}%-{ca.max():.0f}%]")

        metric = val_acc if has_val else train_acc
        if metric > best_acc:
            best_acc = metric
            best_val_acc = val_acc
            best_weights = trainer.weights.clone()
            best_thresholds = trainer.thresholds.clone()
            no_improve = 0
            if save_path:
                trainer.save_model(save_path)
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_weights is not None:
        trainer.weights = best_weights
        trainer.thresholds = best_thresholds
    return trainer


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description='Pruning-oriented SNN Trainer')
    ap.add_argument('--trainer',       choices=['bp', 'stdp'], default='bp')
    ap.add_argument('--neurons',       type=int,   default=15)
    ap.add_argument('--hidden-size',   type=int,   default=1000)
    ap.add_argument('--epochs',        type=int,   default=200)
    ap.add_argument('--batch-size',    type=int,   default=128)
    ap.add_argument('--train-samples', type=int,   default=0)
    ap.add_argument('--test-samples',  type=int,   default=0)
    ap.add_argument('--patience',      type=int,   default=20)
    ap.add_argument('--seed',          type=int,   default=42)
    ap.add_argument('--lr',            type=float, default=1e-3)
    ap.add_argument('--weight-decay',  type=float, default=1e-4)
    ap.add_argument('--timesteps',     type=int,   default=1,
                    help='Deterministic input presentations per sample; current hardware uses 1')
    ap.add_argument('--input-threshold', type=float, default=0.3,
                    help='Emit an input spike when normalized pixel value exceeds this threshold')
    ap.add_argument('--hidden-threshold', type=float, default=1.0)
    ap.add_argument('--output-threshold', type=float, default=1.0)
    ap.add_argument('--leak',          type=float, default=0.0,
                    help='Software membrane leak; 0 matches the current deployment configuration')
    ap.add_argument('--surrogate-scale', type=float, default=10.0)
    ap.add_argument('--enable-alt-mapping-prune', action='store_true',
                    help='Alternate BP training with core-group mapping and cost-aware pruning')
    ap.add_argument('--map-warmup-epochs', type=int, default=5)
    ap.add_argument('--map-interval-epochs', type=int, default=5)
    ap.add_argument('--map-num-groups', type=int, default=16)
    ap.add_argument('--map-neurons-per-group', type=int, default=128)
    ap.add_argument('--map-input-hidden-topk', type=int, default=16)
    ap.add_argument('--map-hidden-output-topk', type=int, default=10)
    ap.add_argument('--map-min-abs-weight', type=float, default=0.0)
    ap.add_argument('--map-load-cap', type=float, default=0.0,
                    help='0 means auto average group load * 1.20')
    ap.add_argument('--map-local-edge-cap', type=int, default=0,
                    help='0 means auto neurons_per_group * max(topk)')
    ap.add_argument('--map-lambda-load', type=float, default=10.0)
    ap.add_argument('--map-lambda-cap', type=float, default=1000.0)
    ap.add_argument('--map-lambda-balance', type=float, default=0.05)
    ap.add_argument('--map-max-iter', type=int, default=0)
    ap.add_argument('--importance-mode', choices=['rate-weight'], default='rate-weight',
                    help='Connection importance proxy for mask update')
    ap.add_argument('--runtime-alpha', type=float, default=1.0,
                    help='Multiplier for mapping-derived runtime cost in top-k mask update')
    ap.add_argument('--runtime-cross-rate-coeff', type=float, default=1.0,
                    help='Cross-group runtime cost coefficient multiplied by source firing rate')
    ap.add_argument('--runtime-load-balance-coeff', type=float, default=1.0,
                    help='Load-balance runtime cost coefficient for L_g - mean(L)')
    ap.add_argument('--runtime-load-reward-cap', type=float, default=0.5,
                    help='Clamp lower bound magnitude for normalized low-load reward')
    ap.add_argument('--runtime-load-penalty-cap', type=float, default=1.0,
                    help='Clamp upper bound for normalized high-load penalty')
    ap.add_argument('--prune-cross-penalty', type=float, default=0.0,
                    help='Deprecated; ignored by current runtime cost model')
    ap.add_argument('--prune-load-penalty', type=float, default=0.0,
                    help='Deprecated; ignored by current runtime cost model')
    ap.add_argument('--mapping-snapshot-prefix', default='',
                    help='Optional prefix for per-mapping .npz/.json snapshots')
    args = ap.parse_args()
    if args.timesteps < 1:
        ap.error('--timesteps must be at least 1')
    if not 0.0 <= args.input_threshold <= 1.0:
        ap.error('--input-threshold must be in [0, 1]')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    max_train = args.train_samples if args.train_samples > 0 else None
    max_test = args.test_samples if args.test_samples > 0 else None

    mode = "BPEventScore" if args.trainer == 'bp' else "FaithfulOnChip"
    print("=" * 70)
    print(f"Pruning SNN Trainer - Mode: {mode}")
    print("=" * 70)
    print(f"  Device:       {DEVICE}")
    print(f"  Input:        deterministic pixel>{args.input_threshold:g}, "
          f"T={args.timesteps}, leak={args.leak:g}")
    if args.trainer == 'bp':
        print(f"  Architecture: 784 -> {args.hidden_size} -> 10")
        print("  Hidden:       event-serial, reset-to-zero, fire-once per sample")
        print("  Event order:  random per sample in training, ascending ID in eval")
        print("  Output:       nonnegative-weight score readout + argmax")
    else:
        n_output = args.neurons * 10
        print(f"  Architecture: 784 -> {n_output} ({args.neurons}/class x 10)")

    # Load data
    print("\n[1] Loading MNIST ...")
    (train_imgs, train_lbls), (test_imgs, test_lbls) = load_mnist(
        max_train=max_train, max_test=max_test)
    print(f"  Train: {len(train_imgs)},  Test: {len(test_imgs)}")

    val_size = min(2000, len(test_imgs))

    if args.trainer == 'bp':
        if args.enable_alt_mapping_prune and coregroup_mapping is None:
            raise RuntimeError("需要 tests/coregroup_mapping.py 才能启用 --enable-alt-mapping-prune")
        print(f"\n[2] Creating {mode} trainer ...")
        trainer = BPPruneTrainer(
            n_input=784, hidden_size=args.hidden_size, n_classes=10,
            timesteps=args.timesteps,
            hidden_threshold=args.hidden_threshold,
            output_threshold=args.output_threshold,
            leak=args.leak, surrogate_scale=args.surrogate_scale,
            input_threshold=args.input_threshold,
            device=DEVICE)

        print("\n[3] Pre-training test ...")
        pre = trainer.test_batch(
            test_imgs[:min(2000, len(test_imgs))],
            test_lbls[:min(2000, len(test_imgs))])
        print(f"  Pre-training score acc: {pre['score_acc']:.1f}%")
        print(f"  Pre-training readout acc: {pre['count_acc']:.1f}%")

        model_path = f'data/cache/bp_prune_model_{args.hidden_size}h_10c.npz'
        print(f"\n[4] BP training ({args.epochs} epochs, {len(train_imgs)} samples) ...")
        if args.enable_alt_mapping_prune:
            print("  Alternating mapping/prune: ENABLED")
            print(f"    warmup={args.map_warmup_epochs} interval={args.map_interval_epochs} "
                  f"groups={args.map_num_groups}x{args.map_neurons_per_group}")
            print(f"    topk input->hidden={args.map_input_hidden_topk} "
                  f"hidden->output={args.map_hidden_output_topk}")
            print(f"    importance={args.importance_mode} runtime_alpha={args.runtime_alpha}")
            print(f"    runtime cost: cross=a_i*{args.runtime_cross_rate_coeff}, "
                  f"load=norm(L_g-meanL)*{args.runtime_load_balance_coeff} "
                  f"clamp=[-{args.runtime_load_reward_cap},+{args.runtime_load_penalty_cap}]")
        t0 = time.time()
        trainer = train_bp_model(
            trainer, train_imgs, train_lbls,
            epochs=args.epochs, batch_size=args.batch_size,
            save_path=model_path,
            val_imgs=test_imgs[:val_size], val_lbls=test_lbls[:val_size],
            patience=args.patience,
            lr=args.lr, weight_decay=args.weight_decay,
            alternating_args=args)
        dt = time.time() - t0
        print(f"  Time: {dt:.0f}s ({dt/60:.1f} min)")

        print(f"\n[5] Final test ({len(test_imgs)} images) ...")
        final = trainer.test_batch(test_imgs, test_lbls)

        print("\n" + "=" * 70)
        print(f"RESULTS - {mode}")
        print("=" * 70)
        print(f"  Architecture:        784 -> {args.hidden_size} -> 10")
        print("  Mode:                event-hidden BP, non-spiking score readout")
        print(f"  Pre score/readout:   {pre['score_acc']:.1f}% / {pre['count_acc']:.1f}%")
        print(f"  Final score acc:     {final['score_acc']:.1f}%")
        print(f"  Final readout acc:   {final['count_acc']:.1f}%")

        if args.enable_alt_mapping_prune:
            int8_ref = trainer.test_int8_reference(test_imgs, test_lbls)
            print("\n  INT8 deployment software reference:")
            print(f"    Accuracy:             {int8_ref['accuracy']:.1f}%")
            print(f"    Hidden spikes/sample: {int8_ref['hidden_spikes_per_sample']:.3f}")
            print(f"    Active readouts/sample:{int8_ref['active_readouts_per_sample']:.3f}")
            print(f"    INT8 scale/threshold: {int8_ref['weight_scale']:.3f}"
                  f"/{int8_ref['hardware_threshold']}")
            print(f"    Matching mapping:     {trainer.matching_mapping}")

        print("\n  Per-class score/readout accuracy:")
        for c in range(10):
            ct = int(final['per_class_t'][c].item())
            sc = int(final['per_class_score_c'][c].item())
            cc = int(final['per_class_count_c'][c].item())
            print(f"    Class {c}: score {sc}/{ct}={sc/max(ct,1)*100:.0f}%  "
                  f"readout {cc}/{ct}={cc/max(ct,1)*100:.0f}%")

        print("\n  Note: this BP model is a training framework for 784->1000->10.")
        print("  Convert with tests/prepare_bp_coregroup_deployment.py before deployment.")
        return final['score_acc']

    # Original faithful STDP route, kept as a baseline.
    n_output = args.neurons * 10
    print(f"\n[2] Creating {mode} trainer ...")
    trainer = FaithfulOnChipTrainer(
        n_input=784, n_classes=10, features_per_class=args.neurons,
        leak=args.leak, timesteps=args.timesteps,
        input_threshold=args.input_threshold,
        lr_plus=0.005, lr_minus=0.003, device=DEVICE)
    trainer.init_prototypes(train_imgs, train_lbls)

    print("\n[3] Pre-training test ...")
    pre_acc, _, _ = trainer.test_batch(
        test_imgs[:min(2000, len(test_imgs))],
        test_lbls[:min(2000, len(test_imgs))])
    print(f"  Pre-training accuracy: {pre_acc:.1f}%")

    model_path = f'data/cache/onchip_prune_model_{n_output}n.npz'
    print(f"\n[4] Training ({args.epochs} epochs, {len(train_imgs)} samples) ...")
    t0 = time.time()
    trainer = train_model(
        trainer, train_imgs, train_lbls,
        epochs=args.epochs, batch_size=args.batch_size,
        save_path=model_path,
        val_imgs=test_imgs[:val_size], val_lbls=test_lbls[:val_size],
        patience=args.patience)
    dt = time.time() - t0
    print(f"  Time: {dt:.0f}s ({dt/60:.1f} min)")

    print(f"\n[5] Final test ({len(test_imgs)} images) ...")
    final_acc, pc_c, pc_t = trainer.test_batch(test_imgs, test_lbls)

    print("\n" + "=" * 70)
    print(f"RESULTS - {mode}")
    print("=" * 70)
    ws = trainer.get_weight_stats()
    print(f"  Architecture:      784 -> {n_output} ({args.neurons}/class x 10)")
    print(f"  Mode:              {mode}")
    print(f"  Pre-training acc:  {pre_acc:.1f}%")
    print(f"  Final test acc:    {final_acc:.1f}%")
    print(f"  Weight mean:       {ws['mean_f']:.3f}")

    print("\n  Per-class accuracy:")
    for c in range(10):
        ct = int(pc_t[c].item())
        cc = int(pc_c[c].item())
        print(f"    Class {c}: {cc}/{ct} = {cc/max(ct,1)*100:.0f}%")

    print("\n  Faithful route only: float R-STDP training, int8-quantized eval/deploy.")
    print("  This file is the pruning-experiment base; add pruning masks here.")
    return final_acc


if __name__ == "__main__":
    main()
