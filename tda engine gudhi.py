"""
tda_engine_gudhi.py
===================
Versión del motor que usa la librería GUDHI para el complejo testigo y la
persistencia, en lugar de la implementación propia.

  pip install gudhi numpy scipy matplotlib

Qué pone GUDHI (en vez de mis funciones):
  - FPS de landmarks          -> gudhi.subsampling.choose_n_farthest_points
  - complejo testigo DÉBIL    -> gudhi.EuclideanWitnessComplex
  - complejo testigo FUERTE   -> gudhi.EuclideanStrongWitnessComplex
  - filtración + persistencia -> SimplexTree.create_simplex_tree(...).persistence()
  - números de Betti          -> SimplexTree.betti_numbers()

Qué sigue en numpy (GUDHI no lo ofrece):
  - local_pca_dim   : dimensión intrínseca local por PCA
  - spherical_kmeans: agrupar direcciones -> ejes (con identificación antipodal)

Nota de parámetros: GUDHI controla la escala con `max_alpha_square` (en unidades
de DISTANCIA AL CUADRADO / relajación), no con el "grado de vecindad" que yo usaba.
Aquí se ofrece `auto_max_alpha_square` para fijarlo a partir de la escala local.
"""
import numpy as np
import gudhi
import gudhi.subsampling
from scipy.spatial import cKDTree


# ----------------------------------------------------------------------
# 1. Landmarks por FPS  (GUDHI)
# ----------------------------------------------------------------------
def landmarks_fps(X, n, seed=0, cap=200000):
    """Selecciona n landmarks por farthest point sampling con GUDHI.
    Para nubes enormes, prerecorta aleatoriamente a `cap` puntos (el FPS es O(n·N))."""
    if len(X) > cap:
        X = X[np.random.default_rng(seed).choice(len(X), cap, replace=False)]
    L = gudhi.subsampling.choose_n_farthest_points(points=X, nb_points=min(n, len(X)))
    return np.asarray(L, dtype=float)


# ----------------------------------------------------------------------
# 2. Escala de filtración automática
# ----------------------------------------------------------------------
def auto_max_alpha_square(landmarks, factor=3.0):
    """max_alpha_square ~ (factor · distancia mediana al vecino más cercano)^2.
    Punto de partida razonable; súbelo si H1/H2 no llegan a formarse."""
    nn = cKDTree(landmarks).query(landmarks, k=2)[0][:, 1]
    return float((factor * np.median(nn)) ** 2)


# ----------------------------------------------------------------------
# 3. Complejo testigo + persistencia  (GUDHI)
# ----------------------------------------------------------------------
def witness_persistence(P, nland=200, nwit=30000, max_alpha_square=None,
                        limit_dimension=3, factor=3.0, coeff=2, strong=False, seed=0):
    """Pipeline completo con GUDHI. Devuelve (SimplexTree ya persistido,
    max_alpha_square usado, landmarks).
      - limit_dimension=3 -> incluye tetraedros, necesario para H2.
      - coeff             -> cuerpo Z/coeff. Usa 2 y 3 para detectar torsión (orientabilidad).
      - strong=True       -> usa el testigo FUERTE relajado (más rápido a filtración alta)."""
    rng = np.random.default_rng(seed)
    land = landmarks_fps(P, min(nland, len(P)), seed=seed)
    W = P if len(P) <= nwit else P[rng.choice(len(P), nwit, replace=False)]
    if max_alpha_square is None:
        max_alpha_square = auto_max_alpha_square(land, factor)

    Cls = gudhi.EuclideanStrongWitnessComplex if strong else gudhi.EuclideanWitnessComplex
    wc = Cls(landmarks=land.tolist(), witnesses=W.tolist())
    st = wc.create_simplex_tree(max_alpha_square=max_alpha_square,
                                limit_dimension=limit_dimension)
    st.persistence(homology_coeff_field=coeff, persistence_dim_max=False)
    return st, max_alpha_square, land


def betti(st):
    """Números de Betti (b0,b1,b2). Requiere haber llamado a persistence() antes."""
    b = list(st.betti_numbers())
    return tuple((b + [0, 0, 0])[:3])


def diagrams(st, maxdim=2):
    """Diagramas por dimensión como arrays (n,2) de (nacimiento, muerte).
    La muerte puede ser inf. El eje está en unidades de alpha^2."""
    return {p: st.persistence_intervals_in_dimension(p) for p in range(maxdim + 1)}


# ----------------------------------------------------------------------
# 4. Geometría local (numpy; GUDHI no lo provee)
# ----------------------------------------------------------------------
def local_pca_dim(points, ref, k=40, thr=0.12):
    """Dimensión local (1/2/3) de cada punto por PCA de sus k vecinos en `ref`."""
    tree = cKDTree(ref)
    _, nb = tree.query(points, k=k)
    dim = np.zeros(len(points)); evr = np.zeros((len(points), 3))
    for i in range(len(points)):
        Q = ref[nb[i]] - ref[nb[i]].mean(0)
        l = np.linalg.svd(Q, compute_uv=False) ** 2; l = l / l.sum()
        evr[i] = l; dim[i] = 1 + (l[1] > thr) + (l[2] > thr)
    return dim, evr


def spherical_kmeans(u, k, iters=60, seed=0):
    """k-means esférico con similitud |u·c| (identifica u con -u). Devuelve (ejes, etiquetas)."""
    rng = np.random.default_rng(seed)
    C = u[rng.choice(len(u), k, replace=False)].copy()
    for _ in range(iters):
        lab = np.abs(u @ C.T).argmax(1); newC = np.zeros_like(C)
        for j in range(k):
            pts = u[lab == j]
            if len(pts) < 3: newC[j] = C[j]; continue
            M = (pts[:, :, None] * pts[:, None, :]).sum(0)
            w, v = np.linalg.eigh(M); newC[j] = v[:, -1]
        if np.allclose(np.abs((newC * C).sum(1)), 1, atol=1e-4): C = newC; break
        C = newC
    return C, np.abs(u @ C.T).argmax(1)


# ----------------------------------------------------------------------
# Validación (ejecuta este archivo directamente para comprobar GUDHI)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    u = rng.normal(0, 1, (8000, 3)); S = u / np.linalg.norm(u, axis=1, keepdims=True)
    st, a, _ = witness_persistence(S, nland=120, nwit=8000, factor=3.0)
    print(f"Esfera (max_alpha_square={a:.3f}) -> Betti {betti(st)}  (esperado (1,0,1))")
    a0, b0 = 2.0, 0.8
    th = rng.uniform(0, 2*np.pi, 8000); ph = rng.uniform(0, 2*np.pi, 8000)
    T = np.column_stack([(a0 + b0*np.cos(ph))*np.cos(th),
                         (a0 + b0*np.cos(ph))*np.sin(th), b0*np.sin(ph)])
    st, a, _ = witness_persistence(T, nland=160, nwit=8000, factor=3.0)
    print(f"Toro   (max_alpha_square={a:.3f}) -> Betti {betti(st)}  (esperado (1,2,1))")
