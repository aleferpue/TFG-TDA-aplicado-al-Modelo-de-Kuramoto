"""
01_exploracion.py
=================
Inspección inicial de la nube puntosTFG.mat (antes de cualquier topología):
forma, rangos por eje, distribución radial (detecta el hueco que separa piezas)
y proyecciones 2D/3D para reconocer la geometría.

Salida: preview.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.spatial import cKDTree

MAT = "/mnt/user-data/uploads/puntosTFG.mat"   # <-- ajusta a tu ruta

# El .mat guarda tres vectores columna x, y, z -> los apilamos en (M,3)
d = loadmat(MAT)
X = np.column_stack([d['x'].ravel(), d['y'].ravel(), d['z'].ravel()]).astype(np.float64)
print("nube:", X.shape)

# --- rangos y estadísticos por eje ---
for i, c in enumerate("xyz"):
    print(f"  {c}: [{X[:,i].min():.3f}, {X[:,i].max():.3f}]  media={X[:,i].mean():.3f}  std={X[:,i].std():.3f}")

# --- distancia al origen: revela si es un cascarón, un volumen, o tiene huecos ---
rn = np.linalg.norm(X, axis=1)
print(f"norma al origen: min={rn.min():.3f} max={rn.max():.3f} media={rn.mean():.3f}")

# --- resolución: distancia al vecino más cercano sobre una submuestra ---
rng = np.random.default_rng(0)
sub = X[rng.choice(len(X), 5000, replace=False)]
nn = cKDTree(sub).query(sub, k=2)[0][:, 1]
print(f"dist. vecino más cercano (submuestra): mediana={np.median(nn):.5f}")

# --- histograma radial (en escala log: hace visible el hueco r in [3.0,3.4]) ---
print("\nperfil radial (nº de puntos por anillo):")
h, edges = np.histogram(rn, bins=40)
for k in range(0, 40, 2):
    print(f"  r in [{edges[k]:.2f},{edges[k+1]:.2f}): {'#'*int(60*h[k]/h.max())} {h[k]}")

# --- figura: 3D + tres proyecciones 2D ---
S = X[rng.choice(len(X), 30000, replace=False)]
fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(141, projection='3d')
ax.scatter(S[:, 0], S[:, 1], S[:, 2], s=1, alpha=0.15)
ax.set_title("3D")
for a, (i, j), t in [(142, (0, 1), "xy"), (143, (0, 2), "xz"), (144, (1, 2), "yz")]:
    ax2 = fig.add_subplot(a)
    ax2.scatter(S[:, i], S[:, j], s=1, alpha=0.1)
    ax2.set_title(t); ax2.set_aspect('equal')
plt.tight_layout()
plt.savefig("preview.png", dpi=90)
print("\nfigura -> preview.png")
