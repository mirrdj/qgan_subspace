# Copyright 2024 PennyLane Team
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

import os
import pickle

import numpy as np
import pennylane.numpy as pnp  # Import PennyLane numpy


def train_log(param, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a") as file:
        file.write(param)


def load_model(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "rb") as qc:
        model = pickle.load(qc)
    return model


def save_model(model_object, file_path):
    """Saves the model (Generator or Discriminator) using pickle.
    Note: Pickling entire PennyLane-dependent objects might be fragile.
    Consider saving parameters (e.g., model_object.params) instead if issues arise.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb+") as file:
        pickle.dump(model_object, file)


def save_fidelity_loss(fidelities_history, losses_history, file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        np.save(f, fidelities_history)
        np.save(f, losses_history)


def save_theta(params: pnp.ndarray, file_path: str):
    """Saves the generator parameters (theta) to a text file.

    Args:
        params (pnp.ndarray): The parameters to save (e.g., gen.params).
        file_path (str): The path to save the parameters.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Convert to standard numpy array for savetxt if it's a PennyLane numpy array
    params_np = params.numpy() if hasattr(params, "numpy") else np.asarray(params)
    np.savetxt(file_path, params_np.flatten())  # Flatten to save as a 1D array or handle shape as needed
