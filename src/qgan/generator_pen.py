import pennylane as qml

from qgan.ansatz import get_ansatz_type_circuit
from qgan.ancilla import get_final_gen_state_for_discriminator

from tools import MomentumOptimizer

class Generator():
    def __init__(self, total_input_state: np.ndarray):
        # Set general used params:
        self.size: int = CFG.system_size + (1 if CFG.extra_ancilla else 0)

        # Set the params, for comparison while loading:
        self.ancilla: bool = CFG.extra_ancilla
        self.ancilla_topology: str = CFG.ancilla_topology  # Type doesn't matter, ancilla always passes through gen
        self.ansatz: str = CFG.gen_ansatz
        self.layers: int = CFG.gen_layers
        self.target_size: int = CFG.system_size
        self.target_hamiltonian: str = CFG.target_hamiltonian
        self.total_input_state: np.ndarray = total_input_state

        self.circuit_fn, count = get_ansatz_type_circuit(self.ansatz, self.size, self.layers)
        self.params = np.random.uniform(0, 2 * np.pi, gate_count)
        self.optimizer = MomentumOptimizer()

    def get_total_gen_state(self) -> np.ndarray:
        """Get the total generator state, including the untouched qubits in front (choi).

        Args:
            total_input_state (np.ndarray): The input state vector.

        Returns:
            np.ndarray: The total generator state vector.
        """
        Untouched_x_G: np.ndarray = np.kron(Identity(CFG.system_size), self.qc.get_mat_rep())

        return np.matmul(Untouched_x_G, self.total_input_state)

    def cost_fn(self):
        pass

    def update_gen(self, dis: Discriminator, final_target_state: np.ndarray):
        """Update the generator parameters (angles) using the optimizer.
        
        Args:
            dis (Discriminator): The discriminator to compute gradients.
            final_target_state (np.ndarray): The target state vector.
        """
        grad_fn = qml.grad(self.cost_fn)
        grads = grad_fn(self.params, final_target_state)
        self.params = self.optimizer.move_in_grad(self.params, grads, "min")

    def _ansatz_circuit_state_prep(self, input_state_vector):
        qml.QubitStateVector(total_input_state, wires=range(self.size))

        @qml.qnode(dev, interface="autograd")
        def apply_circuit():
            self.circuit_fn(self.size, self.layers, self.params)
            return qml.state()

        return apply_circuit