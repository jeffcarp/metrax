# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test utilities for metrax."""

import jax.numpy as jnp
import numpy as np

np.random.seed(42)
BATCHES = 4
BATCH_SIZE = 8
OUTPUT_LABELS = np.random.randint(
    0,
    2,
    size=(BATCHES, BATCH_SIZE),
).astype(np.float32)
OUTPUT_PREDS = np.random.uniform(size=(BATCHES, BATCH_SIZE)).astype(np.float32)
OUTPUT_LABELS_BS1 = np.random.randint(
    0,
    2,
    size=(BATCHES, 1),
).astype(np.float32)
OUTPUT_PREDS_BS1 = np.random.uniform(size=(BATCHES, 1)).astype(np.float32)
SAMPLE_WEIGHTS = np.tile(
    [0.5, 1, 0, 0, 0, 0, 0, 0],
    (BATCHES, 1),
).astype(np.float32)
DEFAULT_SAMPLE_WEIGHTS_10 = jnp.array(
    [0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32
)


def generate_model_outputs(
    num_batches: int, num_features_per_batch: int, random_seed: int = 42
):
  """Generates a tuple of dictionaries for model outputs.

  Args:
    num_batches: The number of batches, determining the length of the output
      tuple.
    num_features_per_batch: The number of features in each batch for logits and
      labels.
    random_seed: Seed for numpy's random number generator for reproducibility.

  Returns:
    A tuple of dictionaries. Each dictionary has 'logits' and 'labels' as keys,
    with JAX numpy arrays as values.
  """
  np.random.seed(random_seed)
  model_outputs = []
  for _ in range(num_batches):
    logits = np.random.uniform(size=(num_features_per_batch,))
    labels = np.random.randint(0, 2, size=(num_features_per_batch,))
    model_outputs.append({
        'logits': jnp.array(logits),
        'labels': jnp.array(labels),
    })
  return tuple(model_outputs)
