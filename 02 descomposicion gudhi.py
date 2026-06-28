"""
02_descomposicion_gudhi.py
=========================
Análisis tipo ciclooctano usando GUDHI:
  (1) Betti GLOBAL (poco informativo para una unión no-variedad)
  (2) DIMENSIÓN LOCAL por PCA (test de no-variedad)
  (3) DESCOMPOSICIÓN en cascarón / interior y Betti de cada pieza con GUDHI
      -> el cascarón resulta ser una ESFERA, Betti (1,0,1).

Requiere: pip install gudhi numpy scipy matplotlib
Usa tda_engine_gudhi.py.
Salida: cyclo_analisis_gudhi.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tda_engine_gudhi import witness_persistence, betti, diagrams, local_pca_dim

MAT = "/mnt/user-data/uploads/puntosTFG.mat"   # <-- ajusta a tu ruta
rng = np.random.default_rng(0)

d = loadmat(MAT)
X = np.column_stack([d['x'].ravel(), d['y'].ravel(), d['z'].ravel()]).astype(np.float64)
r = np.linalg.norm(X, axis=1)

# ----------------------------------------------------------------------
# (1) Betti GLOBAL
# ----------------------------------------------------------------------
st_full, a_full, _ = witness_persistence(X, nland=200, nwit=30000, factor=3.0, limit_dimension=3)
print("GLOBAL: Betti =", betti(st_full), " (poco informativo, como en el ciclooctano)")

# ----------------------------------------------------------------------
# (2) DIMENSIÓN LOCAL
# ----------------------------------------------------------------------
ref = X[rng.choice(len(X), 120000, replace=False)]
Sd = X[rng.choice(len(X), 20000, replace=False)]
dim, _ = local_pca_dim(Sd, ref, k=40)
print("dimensión local (%):", {dd: round((dim == dd).mean()*100, 1) for dd in (1, 2, 3)})

# ----------------------------------------------------------------------
# (3) DESCOMPOSICIÓN + Betti por pieza
# ----------------------------------------------------------------------
shell = X[r > 3.3]
inner = X[(r > 0.25) & (r < 2.95)]
st_s, a_s, _ = witness_persistence(shell, nland=150, nwit=30000, factor=3.0, limit_dimension=3)
st_i, a_i, _ = witness_persistence(inner, nland=220, nwit=30000, factor=3.0, limit_dimension=3)
print("CASCARÓN exterior :", betti(st_s), " -> (1,0,1) = ESFERA")
print("INTERIOR          :", betti(st_i), " (no-variedad: recuento sensible a la escala)")

# ----------------------------------------------------------------------
# Figura
# ----------------------------------------------------------------------
def plot_diagram(ax, st, ms, title):
    """Diagrama de persistencia desde los intervalos de GUDHI (eje en alpha^2)."""
    mx = ms * 1.15
    col = {0: 'tab:blue', 1: 'tab:orange', 2: 'tab:green'}; lab = {0: '$H_0$', 1: '$H_1$', 2: '$H_2$'}
    dg = diagrams(st, maxdim=2)
    for p in range(3):
        a = dg[p]
        if not len(a): continue
        ax.scatter(a[:, 0], np.where(np.isfinite(a[:, 1]), a[:, 1], mx),
                   s=16, color=col[p], label=lab[p], alpha=.75)
    ax.plot([0, mx], [0, mx], 'k--', lw=.8); ax.axhline(mx, ls=':', color='gray', lw=.7)
    ax.set_title(title); ax.set_xlabel("nacimiento ($\\alpha^2$)"); ax.set_ylabel("muerte"); ax.legend(fontsize=8)

inner_ref = inner[rng.choice(len(inner), 100000, replace=False)] if len(inner) > 100000 else inner
Si = inner[rng.choice(len(inner), 15000, replace=False)]
ldi, _ = local_pca_dim(Si, inner_ref, k=40)

Sv = X[rng.choice(len(X), 25000, replace=False)]; rv = np.linalg.norm(Sv, axis=1)
fig = plt.figure(figsize=(15, 9))
ax = fig.add_subplot(231, projection='3d')
ax.scatter(*Sv[rv > 3.3].T, s=2, alpha=.25, color='tab:green', label='cascarón (esfera)')
mi = (rv > 0.25) & (rv < 2.95)
ax.scatter(*Sv[mi].T, s=2, alpha=.25, color='tab:purple', label='interior')
ax.set_title("Descomposición"); ax.legend(fontsize=8)
cmap = {1: 'tab:red', 2: 'tab:green', 3: 'tab:blue'}
ax2 = fig.add_subplot(232, projection='3d')
for dd in (2, 1, 3):
    mm = ldi == dd; ax2.scatter(*Si[mm].T, s=2, alpha=.3, color=cmap[dd], label=f"{dd}D")
ax2.set_title("Interior: dimensión local"); ax2.legend(fontsize=8)
ax3 = fig.add_subplot(233)
for dd in (2, 1, 3):
    mm = ldi == dd; ax3.scatter(Si[mm, 0], Si[mm, 1], s=2, alpha=.25, color=cmap[dd])
ax3.set_title("Interior xy"); ax3.set_aspect('equal')
plot_diagram(fig.add_subplot(234), st_s, a_s, "CASCARÓN -> Betti (1,0,1) = esfera")
plot_diagram(fig.add_subplot(235), st_i, a_i, "INTERIOR")
ax6 = fig.add_subplot(236)
hist, edg = np.histogram(r, bins=60); ax6.plot((edg[:-1]+edg[1:])/2, hist)
ax6.axvspan(3.0, 3.4, color='red', alpha=.15, label='hueco radial')
ax6.set_yscale('log'); ax6.set_xlabel("r"); ax6.set_title("Distribución radial"); ax6.legend(fontsize=8)
plt.tight_layout(); plt.savefig("cyclo_analisis_gudhi.png", dpi=105)
print("\nfigura -> cyclo_analisis_gudhi.png")
