import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector


# -----------------
#  Data
# -----------------
X, y = make_moons(n_samples=60, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
num_features = X.shape[1]

# -----------------
#  Feature map
# -----------------
feature_map = zz_feature_map(num_features, reps=2, entanglement="full")

def encode(feature_map, x):
    """Return a feature-map circuit with data encoded."""
    return feature_map.assign_parameters(x, inplace=False)


# -----------------
#  Fidelity kernel
# -----------------


def fidelity_kernel(XA, XB, feature_map):
    states_A = [Statevector(encode(feature_map, a)) for a in XA]
    states_B = [Statevector(encode(feature_map, b)) for b in XB]

    K = np.zeros((len(XA), len(XB)))
    for i, sa in enumerate(states_A):
        for j, sb in enumerate(states_B):
            K[i, j] = np.abs(sa.data.conj() @ sb.data) ** 2
    return K


# -----------------
#  Train + Test
# -----------------
print("Computing quantum kernel (shot-based)...")
K_train = fidelity_kernel(X_train, X_train, feature_map)
K_test = fidelity_kernel(X_test, X_train, feature_map)

clf = SVC(kernel="precomputed")
clf.fit(K_train, y_train)
acc = clf.score(K_test, y_test)

print(f"Test accuracy (shots=1024): {acc:.3f}")
