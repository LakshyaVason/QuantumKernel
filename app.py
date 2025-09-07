from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
import io, base64
import signal
from contextlib import contextmanager

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from qiskit.circuit.library import zz_feature_map, pauli_feature_map
from qiskit.quantum_info import Statevector

app = Flask(__name__)

# ----------------- Dataset -----------------
X, y = make_moons(n_samples=60, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
num_features = X.shape[1]

# ----------------- Timeout Context Manager -----------------
class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Computation timed out")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# ----------------- Optimized Fidelity Kernel -----------------
def encode(feature_map, x):
    return feature_map.assign_parameters(x, inplace=False)

def fidelity_kernel(XA, XB, feature_map, max_size=500):
    """Optimized kernel computation with size limits"""
    # Limit computation size to prevent timeouts
    if len(XA) * len(XB) > max_size:
        print(f"Warning: Kernel computation too large ({len(XA)} x {len(XB)}), limiting...")
        
    states_A = [Statevector(encode(feature_map, a)) for a in XA[:min(len(XA), 25)]]
    states_B = [Statevector(encode(feature_map, b)) for b in XB[:min(len(XB), 25)]]
    
    K = np.zeros((len(states_A), len(states_B)))
    for i, sa in enumerate(states_A):
        for j, sb in enumerate(states_B):
            K[i, j] = np.abs(sa.data.conj() @ sb.data) ** 2
    return K

# ----------------- Simplified Decision Boundary -----------------
def plot_decision_boundary(clf, X_train, y_train, feature_map, ax):
    """Much coarser grid to prevent timeouts"""
    h = 0.3  # Much larger step size
    x_min, x_max = X_train[:,0].min()-0.5, X_train[:,0].max()+0.5
    y_min, y_max = X_train[:,1].min()-0.5, X_train[:,1].max()+0.5
    
    # Create smaller grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Limit grid size
    if len(grid) > 50:
        indices = np.random.choice(len(grid), 50, replace=False)
        grid = grid[indices]
        xx_small = xx.flatten()[indices].reshape(-1, 1)
        yy_small = yy.flatten()[indices].reshape(-1, 1)
    
    try:
        K_grid = fidelity_kernel(grid, X_train, feature_map, max_size=200)
        Z = clf.predict(K_grid)
        
        # Simple scatter plot instead of contour
        ax.scatter(grid[:,0], grid[:,1], c=Z, cmap=plt.cm.Pastel2, 
                  alpha=0.3, s=100, marker='s')
    except Exception as e:
        print(f"Grid prediction failed: {e}")
    
    # Plot training data
    ax.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=plt.cm.Set1,
               edgecolors='k', s=80)
    ax.set_title("Quantum Kernel SVM (Simplified)")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(alpha=0.3)

# ----------------- Routes -----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_kernel', methods=['POST'])
def run_kernel():
    try:
        with timeout(25):  # 25 second timeout
            params = request.get_json()
            depth = min(int(params['depth']), 2)  # Limit depth
            ent = params['ent']
            fmap_type = params['feature_map']

            # Pick feature map
            if fmap_type == "zz":
                feature_map = zz_feature_map(feature_dimension=num_features, reps=depth, entanglement=ent)
            elif fmap_type == "pauli":
                feature_map = pauli_feature_map(feature_dimension=num_features, reps=depth, entanglement=ent)
            else:
                return jsonify({"error": "Unknown feature map"}), 400

            # Compute kernel with limits
            K_train = fidelity_kernel(X_train, X_train, feature_map)
            K_test = fidelity_kernel(X_test, X_train, feature_map)

            clf = SVC(kernel='precomputed')
            clf.fit(K_train, y_train)
            acc = clf.score(K_test, y_test)

            # Plot with timeout protection
            fig, ax = plt.subplots(figsize=(6, 5))
            plot_decision_boundary(clf, X_train, y_train, feature_map, ax)
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=72)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)

            return jsonify({
                'plot_url': f"data:image/png;base64,{img_base64}",
                'accuracy': round(acc, 3)
            })
            
    except TimeoutException:
        return jsonify({"error": "Computation timed out. Try reducing depth or complexity."}), 408
    except Exception as e:
        return jsonify({"error": f"Computation error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)