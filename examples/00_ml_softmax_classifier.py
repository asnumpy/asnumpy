# *****************************************************************************
# Copyright (c) 2025 AISS and ISE Group at Harbin Institute of Technology.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *****************************************************************************

"""
ML Softmax Classifier -- AsNumpy Comprehensive Demo
=========================================

Forward inference of a 3-layer FC network using AsNumpy,
simulating MNIST handwritten digit classification (inference only, no training):

    Input(784) -> Dense(256, ReLU) -> Dense(128, GELU) -> Dense(10, Softmax) -> Prediction
"""

import gc
import os
import sys
import time

import numpy as np
from loguru import logger
from scipy.special import erf, softmax as sp_softmax

import asnumpy as ap
from utils import calculate_stable_metric

# ─── Network hyperparameters ───────────────────────────

INPUT_DIM = 784       # MNIST image: 28x28
HIDDEN1 = 256
HIDDEN2 = 128
NUM_CLASSES = 10
BATCH_SIZE = 128
DTYPE = np.float32


# ─── Utility functions ───────────────────────────


def he_init(fan_in: int, fan_out: int) -> np.ndarray:
    """He initialization: weights ~ N(0, sqrt(2/fan_in))."""
    std = np.sqrt(2.0 / fan_in)
    return np.random.normal(0, std, (fan_in, fan_out)).astype(DTYPE)


def make_onehot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer labels to one-hot encoding."""
    onehot = np.zeros((len(labels), num_classes), dtype=DTYPE)
    onehot[np.arange(len(labels)), labels] = 1.0
    return onehot


def np_reference_forward(x_np, W1, W2, W3, b1, b2, b3):
    """Pure NumPy reference forward pass for verification."""
    z1 = np.add(np.dot(x_np, W1), b1)
    h1 = np.maximum(z1, 0)                                    # ReLU
    z2 = np.add(np.dot(h1, W2), b2)
    h2 = z2 * 0.5 * (1 + erf(z2 / np.sqrt(2.0)))            # GELU
    logits = np.add(np.dot(h2, W3), b3)
    probs = sp_softmax(logits, axis=-1)
    return probs


# ─── Stage 1: Data and weight preparation ─────────────


def prepare_model():
    """
    Generate model weights and test data, upload to NPU.
    """
    logger.info("=" * 60)
    logger.info("Stage 1: Preparing model weights and test data")
    logger.info("=" * 60)

    # Weight matrices: NumPy generation (He init) -> upload to NPU
    layer_configs = [
        (INPUT_DIM, HIDDEN1),   # W1
        (HIDDEN1, HIDDEN2),     # W2
        (HIDDEN2, NUM_CLASSES), # W3
    ]
    W_np = [he_init(fin, fout) for fin, fout in layer_configs]
    W_npu = [ap.ndarray.from_numpy(w) for w in W_np]

    # Biases: directly create on NPU (no CPU round-trip needed)
    bias_sizes = [HIDDEN1, HIDDEN2, NUM_CLASSES]
    b_npu = [ap.zeros(size, dtype=DTYPE) for size in bias_sizes]

    for i, ((fin, fout), w) in enumerate(zip(layer_configs, W_npu)):
        logger.info(f"  W{i+1}: {fin}x{fout}")

    # Simulated MNIST batch
    x_np = np.random.normal(0, 0.5, (BATCH_SIZE, INPUT_DIM)).astype(DTYPE)
    x = ap.ndarray.from_numpy(x_np)
    logger.info(f"  Input batch: ({BATCH_SIZE}, {INPUT_DIM})")

    # Ground truth labels + one-hot
    labels_np = np.random.randint(0, NUM_CLASSES, size=BATCH_SIZE)
    labels_onehot = ap.ndarray.from_numpy(
        make_onehot(labels_np, NUM_CLASSES)
    )

    # Save/load round-trip verification
    save_path = "/tmp/asnumpy_model_weights.npz"
    ap.savez(save_path, W1=W_npu[0], W2=W_npu[1], W3=W_npu[2],
              b1=b_npu[0], b2=b_npu[1], b3=b_npu[2])
    file_size = os.path.getsize(save_path) / (1024 * 1024)
    logger.info(f"  Weights saved to {save_path} ({file_size:.2f} MB)")

    loaded = ap.load(save_path)
    diff_norm = np.max(np.abs(W_np[0] - loaded["W1"].to_numpy()))
    logger.info(f"  Save/Load round-trip max |W1 - loaded|: {diff_norm:.2e}")
    del loaded

    return W_npu, b_npu, x, labels_np, labels_onehot, save_path


# ─── Stage 2: Forward inference ──────────────────────


def forward(x, W_npu, b_npu):
    """
    3-layer fully connected forward pass on NPU.
    """
    z1 = ap.add(ap.dot(x, W_npu[0]), b_npu[0])
    h1 = ap.relu(z1)
    z2 = ap.add(ap.dot(h1, W_npu[1]), b_npu[1])
    h2 = ap.gelu(z2)
    logits = ap.add(ap.dot(h2, W_npu[2]), b_npu[2])
    return ap.softmax(logits, axis=-1)


# ─── Stage 3: Result analysis ────────────────────────


def analyze_results(probs, labels_np, labels_onehot):
    """
    Analyze inference results: accuracy, confidence distribution, cross-entropy loss.
    """
    logger.info("=" * 60)
    logger.info("Stage 3: Result analysis")
    logger.info("=" * 60)

    probs_np = probs.to_numpy()

    # argmax on CPU (NPU has no argmax API)
    predictions = np.argmax(probs_np, axis=1)

    # Accuracy
    correct = np.sum(predictions == labels_np)
    accuracy = correct / len(labels_np) * 100
    logger.info(f"  Accuracy: {correct}/{len(labels_np)} = {accuracy:.1f}%")

    # Confidence stats: extract max per row on NPU, then sort on NPU to demo ap.sort
    confidences = ap.max(probs, axis=1)
    confidences_np = confidences.to_numpy()
    sorted_conf = ap.sort(confidences).to_numpy()

    logger.info(f"  Confidence: mean={np.mean(confidences_np):.4f}, "
                f"std={np.sqrt(np.var(confidences_np)):.4f}, "
                f"min={np.min(confidences_np):.4f}, max={np.max(confidences_np):.4f}")
    logger.info(f"  Top-5 confidences: {sorted_conf[-5:][::-1]}")

    # High-confidence predictions
    high_conf = np.sum(confidences_np > 0.5)
    logger.info(f"  High-confidence (>0.5): {high_conf}/{len(labels_np)}")

    # Softmax sum verification (should equal 1.0 per row)
    prob_sums = ap.sum(probs, axis=1).to_numpy()
    logger.info(f"  Softmax sum check (max |sum - 1|): {np.max(np.abs(prob_sums - 1.0)):.2e}")

    # Cross-entropy loss: H(p, q) = -sum(p * log(q)) / N
    epsilon = np.finfo(np.float32).eps
    clipped = ap.maximum(probs, ap.full(probs.shape, epsilon, dtype=DTYPE))
    log_probs = ap.log(clipped)
    elementwise_loss = ap.multiply(labels_onehot, ap.negative(log_probs))
    ce_total = ap.sum(elementwise_loss)
    # ap.sum with axis=None returns a scalar, so division is done on CPU
    ce_per_sample = float(ce_total) / BATCH_SIZE
    logger.info(f"  Cross-entropy loss: {ce_per_sample:.4f}")

    del clipped, log_probs, elementwise_loss, ce_total

    return accuracy


# ─── Stage 4: Performance benchmark ───────────────────


def benchmark_inference(x, W_npu, b_npu, weights_np, warmup=10, iterations=100):
    """
    Benchmark full forward pass at different batch sizes.
    """
    logger.info("=" * 60)
    logger.info("Stage 4: Performance benchmark")
    logger.info("=" * 60)

    batch_sizes = [32, 64, 128, 256, 512]

    print(f"\n{'Batch Size':<12} | {'AsNumpy':<14} | {'NumPy':<14} | {'Speedup':<10}")
    print(f"{'':12} | {'(ms)':<14} | {'(ms)':<14} | {'':10}")
    print("-" * 60)

    results = []
    for bs in batch_sizes:
        x_np = np.random.normal(0, 0.5, (bs, INPUT_DIM)).astype(DTYPE)
        x_npu = ap.ndarray.from_numpy(x_np)

        # AsNumpy benchmark
        asnp_times = []
        for _ in range(warmup):
            _ = forward(x_npu, W_npu, b_npu)
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = forward(x_npu, W_npu, b_npu)
            t1 = time.perf_counter()
            asnp_times.append(t1 - t0)

        # NumPy benchmark
        np_times = []
        for _ in range(warmup):
            _ = np_reference_forward(x_np, *weights_np)
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = np_reference_forward(x_np, *weights_np)
            t1 = time.perf_counter()
            np_times.append(t1 - t0)

        asnp_best = calculate_stable_metric(asnp_times) * 1000
        np_best = calculate_stable_metric(np_times) * 1000
        speedup = np_best / asnp_best if asnp_best > 0 else 0

        print(f"{bs:<12} | {asnp_best:<14.4f} | {np_best:<14.4f} | {speedup:<10.2f}x")
        results.append((bs, asnp_best, np_best, speedup))

        del x_npu, x_np
        gc.collect()

    print("-" * 60)

    if results:
        avg_speedup = sum(r[3] for r in results) / len(results)
        best_bs, _, _, best_speedup = max(results, key=lambda r: r[3])
        logger.info(f"  Average speedup: {avg_speedup:.2f}x")
        logger.info(f"  Best speedup:    {best_speedup:.2f}x (batch_size={best_bs})")

    return results


# ─── Main ──────────────────────────────


class _SuppressCStdout:
    """Redirect C-level stdout (fd 1) to /dev/null while keeping Python
    print() visible via the original file descriptor.
    C++ spdlog writes to fd 1 directly, so dup2 silences it;
    Python print() goes through sys.stdout which we reroute to the saved fd."""

    def __enter__(self):
        self._saved_fd = os.dup(1)
        self._null_fd = os.open(os.devnull, os.O_WRONLY, 0o644)
        os.dup2(self._null_fd, 1)
        self._saved_stdout = sys.stdout
        sys.stdout = open(self._saved_fd, "w", closefd=False, buffering=1)
        return self

    def __exit__(self, *args):
        try:
            sys.stdout.flush()
        except Exception:
            pass
        sys.stdout.close()
        sys.stdout = self._saved_stdout
        os.dup2(self._saved_fd, 1)
        os.close(self._saved_fd)
        os.close(self._null_fd)


def main():
    with _SuppressCStdout():
        _run()


def _run():
    logger.info("=" * 70)
    logger.info("AsNumpy Demo: 3-Layer Fully Connected Softmax Classifier")
    logger.info("Architecture: Input(784) -> Dense(256, ReLU) -> Dense(128, GELU)")
    logger.info("             -> Dense(10, Softmax) -> Prediction")
    logger.info("=" * 70)

    # Stage 1
    W_npu, b_npu, x, labels_np, labels_onehot, save_path = prepare_model()

    try:
        # Pre-compute weights_np once (avoid repeated D2H in benchmark)
        weights_np = [w.to_numpy() for w in W_npu] + [b.to_numpy() for b in b_npu]

        # Stage 2
        logger.info("\n" + "=" * 60)
        logger.info("Stage 2: Forward inference on NPU")
        logger.info("=" * 60)

        probs = forward(x, W_npu, b_npu)
        logger.info(f"  Output shape: {probs.shape}, dtype: {probs.dtype}")

        # Verify against NumPy reference
        x_np = x.to_numpy()
        probs_ref = np_reference_forward(x_np, *weights_np)
        max_diff = np.max(np.abs(probs.to_numpy() - probs_ref))
        logger.info(f"  Max difference vs NumPy: {max_diff:.2e} "
                     f"({'PASSED' if max_diff < 1e-3 else 'FAILED'})")

        # Stage 3
        analyze_results(probs, labels_np, labels_onehot)

        # Stage 4
        benchmark_inference(x, W_npu, b_npu, weights_np)

    finally:
        del W_npu, b_npu, x, probs, labels_onehot
        gc.collect()
        try:
            os.remove(save_path)
        except FileNotFoundError:
            pass
        logger.info(f"\n  Cleaned up {save_path}")


if __name__ == "__main__":
    main()
