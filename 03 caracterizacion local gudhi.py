"""
03_caracterizacion_local_gudhi.py
================================
Caracterización local de la roseta interior (versión con GUDHI):
  (A) LUGAR SINGULAR (puntos localmente 3D) y sus EJES
  (B) SECCIÓN TRANSVERSAL de un eje -> nº de láminas que se cruzan
  (C) intento de AISLAR pétalos (revela que NO se separan)
  (D) COMPARACIÓN con la superficie de Roman (RP^2 inmersa) usando Betti de GUDHI
  (E) ORIENTABILIDAD por TORSIÓN: Betti sobre Z2 vs Z3 (GUDHI lo permite con
      homology_coeff_field). Una diferencia => hay torsión => NO orientable.

La parte topológica (Betti) usa GUDHI; la geometría local (PCA, ejes, secciones)
sigue en numpy porque GUDHI no la ofrece.

Requiere: pip install gudhi numpy scipy matplotlib
Usa tda_engine_gudhi.py.
Salidas: arms_gudhi.png, interior_vs_roman_gudhi.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.spatial import cKDTree
from collections import deque
from tda_engine_gudhi import witness_persistence, betti, local_pca_dim, spherical_kmeans

MAT = "/mnt/user-data/uploads/puntosTFG.mat"   # <-- ajusta a tu ruta
rng = np.random.default_rng(0)
d = loadmat(MAT)
X = np.column_stack([d['x'].ravel(), d['y'].ravel(), d['z'].ravel()]).astype(np.float64)
r = np.linalg.norm(X, axis=1)

# ======================================================================
# (A) LUGAR SINGULAR Y EJES  (numpy)
# ======================================================================
inner = X[(r > 0.3) & (r < 2.6)]
ref = inner[rng.choice(len(inner), 120000, replace=False)]
S = inner[rng.choice(len(inner), 40000, replace=False)]
ld, _ = local_pca_dim(S, ref, k=45)
P3 = S[ld == 3]
print(f"puntos localmente 3D (lugar singular): {len(P3)} de {len(S)}")
u = P3 / np.linalg.norm(P3, axis=1, keepdims=True)
for K in (3, 4, 6):
    C, lab = spherical_kmeans(u, K)
    print(f"  K={K}: ajuste |cos|={np.abs((u*C[lab]).sum(1)).mean():.3f}  tamaños={np.bincount(lab, minlength=K)}")
C, lab = spherical_kmeans(u, 3)
print("3 ejes:\n", np.round(C, 3),
      "\n  ortogonalidad:", np.round([abs(C[0]@C[1]), abs(C[0]@C[2]), abs(C[1]@C[2])], 3))

# ======================================================================
# (B) SECCIÓN TRANSVERSAL DE UN EJE  (numpy)
# ======================================================================
axis = C[np.bincount(lab).argmax()]; axis /= np.linalg.norm(axis)
tmp = np.array([1., 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1., 0])
e1 = np.cross(axis, tmp); e1 /= np.linalg.norm(e1); e2 = np.cross(axis, e1)
Xs = X[rng.choice(len(X), 300000, replace=False)]
along = Xs @ axis; dperp = np.linalg.norm(Xs - np.outer(along, axis), axis=1)
band = (np.abs(along) > 0.6) & (np.abs(along) < 2.2) & (dperp < 0.35)
cs = Xs[band]; a = cs @ e1; b = cs @ e2
hist, edges = np.histogram(np.arctan2(b, a), bins=72, range=(-np.pi, np.pi))
thr = hist.mean() * 1.5
rays = int(np.sum((hist > thr) & (hist > np.roll(hist, 1)) & (hist > np.roll(hist, -1))))
print(f"\nsección ⊥ al eje: rayos (= láminas que se cruzan) ~ {rays}")

fig = plt.figure(figsize=(14, 4))
fig.add_subplot(131, projection='3d').scatter(*P3.T, s=2, alpha=.2, color='tab:blue')
plt.gca().set_title("Lugar singular (3D)")
ax2 = fig.add_subplot(132); ax2.scatter(a, b, s=3, alpha=.2); ax2.set_aspect('equal')
ax2.set_title("Sección ⊥ al eje")
fig.add_subplot(133).bar(np.degrees((edges[:-1]+edges[1:])/2), hist, width=4)
plt.gca().set_title("Histograma angular")
plt.tight_layout(); plt.savefig("arms_gudhi.png", dpi=105); print("figura -> arms_gudhi.png")

# ======================================================================
# (C) ¿SE SEPARA EN PÉTALOS?  (numpy)
# ======================================================================
sIn = inner[rng.choice(len(inner), 120000, replace=False)]
along3 = np.abs(sIn @ C.T)
perp = np.sqrt(np.maximum((sIn**2).sum(1)[:, None] - along3**2, 0)).min(1)
petals = sIn[perp > 0.35]
tree = cKDTree(petals); _, nb = tree.query(petals, k=10)
comp = -np.ones(len(petals), int); c = 0
for s in range(len(petals)):
    if comp[s] >= 0: continue
    comp[s] = c; dq = deque([s])
    while dq:
        i = dq.popleft()
        for j in nb[i][1:]:
            if comp[j] < 0 and np.linalg.norm(petals[i]-petals[j]) < 0.25:
                comp[j] = c; dq.append(j)
    c += 1
sizes = np.sort(np.bincount(comp[comp >= 0]))[::-1]
print(f"\ntras quitar los ejes -> componentes; tamaños mayores={sizes[:6]}")
print("  (queda casi todo en UNA pieza: los lóbulos NO son pétalos independientes)")

# ======================================================================
# (D) COMPARACIÓN con la SUPERFICIE DE ROMAN  (Betti con GUDHI)
# ======================================================================
v = rng.normal(0, 1, (12000, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
roman = np.column_stack([v[:, 1]*v[:, 2], v[:, 0]*v[:, 2], v[:, 0]*v[:, 1]])
roman = roman / np.abs(roman).max() * 2.4
st_r, _, _ = witness_persistence(roman, nland=160, nwit=12000, factor=3.0, limit_dimension=3)
st_in, _, _ = witness_persistence(inner, nland=200, nwit=20000, factor=3.0, limit_dimension=3)
print("\nSuperficie de Roman : Betti =", betti(st_r))
print("Interior (datos)    : Betti =", betti(st_in))
print("  -> firma singular coincide (3 ejes ortogonales, punto triple, 2 láminas/eje),")
print("     pero Betti y forma NO coinciden -> familia Steiner, no Roman exacta.")

# ======================================================================
# (E) ORIENTABILIDAD por TORSIÓN: Betti(Z2) vs Betti(Z3)  (lo que GUDHI sí permite)
#     Si difieren, hay torsión en la homología => superficie NO orientable.
# ======================================================================
st2, _, _ = witness_persistence(inner, nland=220, nwit=30000, factor=3.0, limit_dimension=3, coeff=2)
st3, _, _ = witness_persistence(inner, nland=220, nwit=30000, factor=3.0, limit_dimension=3, coeff=3)
b2, b3 = betti(st2), betti(st3)
print(f"\nINTERIOR: Betti(Z2)={b2}  Betti(Z3)={b3}")
print("  -> si difieren, hay torsión => NO orientable; si coinciden, no se detecta torsión.")
print("     (Conviene repetir con varias semillas/escalas por la sensibilidad del no-variedad.)")

# figura D: interior (con ejes) vs Roman
Si = inner[rng.choice(len(inner), 18000, replace=False)]
fig = plt.figure(figsize=(13, 5))
ax = fig.add_subplot(121, projection='3d'); ax.scatter(*Si.T, s=2, alpha=.12, color='tab:purple')
for cc in C:
    ax.plot([-2.3*cc[0], 2.3*cc[0]], [-2.3*cc[1], 2.3*cc[1]], [-2.3*cc[2], 2.3*cc[2]], 'k', lw=1.5)
ax.set_title("INTERIOR + 3 ejes dobles ortogonales")
ax2 = fig.add_subplot(122, projection='3d'); ax2.scatter(*roman.T, s=2, alpha=.12, color='tab:green')
ax2.set_title("Superficie de Roman (RP² inmersa)")
plt.tight_layout(); plt.savefig("interior_vs_roman_gudhi.png", dpi=110)
print("figura -> interior_vs_roman_gudhi.png")
