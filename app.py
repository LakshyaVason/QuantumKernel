from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
import io, base64

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

# ----------------- Fidelity Kernel -----------------
def encode(feature_map, x):
    return feature_map.assign_parameters(x, inplace=False)

def fidelity_kernel(XA, XB, feature_map):
    states_A = [Statevector(encode(feature_map, a)) for a in XA]
    states_B = [Statevector(encode(feature_map, b)) for b in XB]
    K = np.zeros((len(XA), len(XB)))
    for i, sa in enumerate(states_A):
        for j, sb in enumerate(states_B):
            K[i, j] = np.abs(sa.data.conj() @ sb.data) ** 2
    return K

# ----------------- Decision Boundary -----------------
def plot_decision_boundary(clf, X_train, y_train, feature_map, ax):
    h = 0.05
    x_min, x_max = X_train[:,0].min()-0.5, X_train[:,0].max()+0.5
    y_min, y_max = X_train[:,1].min()-0.5, X_train[:,1].max()+0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    K_grid = fidelity_kernel(grid, X_train, feature_map)
    Z = clf.predict(K_grid).reshape(xx.shape)

    ax.contourf(xx, yy, Z, cmap=plt.cm.Pastel2, alpha=0.6)
    ax.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=plt.cm.Set1,
               edgecolors='k', s=80)
    ax.set_title("Quantum Kernel SVM")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(alpha=0.3)

# ----------------- Routes -----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_kernel', methods=['POST'])
def run_kernel():
    params = request.get_json()
    depth = int(params['depth'])
    ent = params['ent']
    fmap_type = params['feature_map']

    # ✅ dynamically pick feature map
    if fmap_type == "zz":
        feature_map = zz_feature_map(feature_dimension=num_features, reps=depth, entanglement=ent)
    elif fmap_type == "pauli":
        feature_map = pauli_feature_map(feature_dimension=num_features, reps=depth, entanglement=ent)
    else:
        return jsonify({"error": "Unknown feature map"}), 400

    # Compute kernel
    K_train = fidelity_kernel(X_train, X_train, feature_map)
    K_test = fidelity_kernel(X_test, X_train, feature_map)

    clf = SVC(kernel='precomputed')
    clf.fit(K_train, y_train)
    acc = clf.score(K_test, y_test)

    # ---- Plot decision boundary ----
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_decision_boundary(clf, X_train, y_train, feature_map, ax)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    return jsonify({
        'plot_url': f"data:image/png;base64,{img_base64}",
        'accuracy': round(acc, 3)
    })

if __name__ == '__main__':
    app.run(debug=True)
