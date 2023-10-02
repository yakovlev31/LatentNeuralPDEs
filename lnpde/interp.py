import numpy as np
from fenics import FunctionSpace, Mesh, MeshEditor, Point, cells, dof_to_vertex_map
import torch
from matplotlib.tri import Triangulation
from scipy.sparse import coo_matrix, lil_matrix
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay


ndarray = np.ndarray
Tensor = torch.Tensor

OUT_OF_DOMAIN_NODE_VALUE = -1.0
SPATIAL_TOL = 1e-6


def get_circle_points():
    points = [
        [np.cos(0), np.sin(0)],
        [np.cos(np.pi/4), np.sin(np.pi/4)],
        [np.cos(np.pi/2), np.sin(np.pi/2)],
        [np.cos(3*np.pi/4), np.sin(3*np.pi/4)],
        [np.cos(np.pi), np.sin(np.pi)],
        [np.cos(5*np.pi/4), np.sin(5*np.pi/4)],
        [np.cos(3*np.pi/2), np.sin(3*np.pi/2)],
        [np.cos(7*np.pi/4), np.sin(7*np.pi/4)],
    ]
    return np.stack(points)


def create_n_circle_stencil(n: int) -> ndarray:
    points = []
    circle_points = get_circle_points()
    grid = np.linspace(0, 1, n + 1)[1:]
    for circle_radius in grid:
        points.extend(circle_points * circle_radius)
    return np.stack(points)


def construct_stencil(stencil_shape: str, stencil_size: float) -> ndarray:
    unit_stencil = {
        "2cross": np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=np.float32),
        "2square": np.array([[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]], dtype=np.float32),
        "4cross": np.array(
            [
                [0, 0.5], [0.5, 0], [0, -0.5], [-0.5, 0],
                [0, 1], [1, 0], [0, -1], [-1, 0]
            ],
            dtype=np.float32
        ),
        "4square": np.array(
            [
                [0, 0.5], [0.5, 0.5], [0.5, 0], [0.5, -0.5], [0, -0.5], [-0.5, -0.5], [-0.5, 0], [-0.5, 0.5],
                [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1],
            ],
            dtype=np.float32
        ),
        "1circle": create_n_circle_stencil(1).astype(np.float32),
        "2circle": create_n_circle_stencil(2).astype(np.float32),
        "3circle": create_n_circle_stencil(3).astype(np.float32),
        "4circle": create_n_circle_stencil(4).astype(np.float32),
        "5circle": create_n_circle_stencil(5).astype(np.float32),
        "6circle": create_n_circle_stencil(6).astype(np.float32),
        "7circle": create_n_circle_stencil(7).astype(np.float32),
        "8circle": create_n_circle_stencil(8).astype(np.float32),
    }
    return unit_stencil[stencil_shape] * stencil_size


def get_neighb_points(x: ndarray, stencil: ndarray) -> ndarray:
    x_neighb = []
    for xi in x:
        x_neighb.extend(xi + stencil)
    return np.stack(x_neighb)


def is_point_inside_rectangle(point: ndarray, bounds: ndarray) -> bool:
    is_inside = True
    if point[0] < bounds[0, 0]:
        is_inside = False
    elif point[0] > bounds[0, 1]:
        is_inside = False
    elif point[1] < bounds[1, 0]:
        is_inside = False
    elif point[1] > bounds[1, 1]:
        is_inside = False
    return is_inside


def apply_periodic_domain(x: ndarray, x_neighb: ndarray) -> ndarray:
    """Maps neighborhood points back to the rectangular domain.

    Args:
        x: Domain points. Used to compute domain boundaries. Has shape (nodes, 2).
        x_neighb: Neighborhood points. Has shape (N_eval, 2).

    Returns:
        Adjusted neighborhood points.
    """
    x_neighb = np.copy(x_neighb)
    x_neighb[:, 0] = x_neighb[:, 0] % x[:, 0].max()
    x_neighb[:, 1] = x_neighb[:, 1] % x[:, 1].max()
    return x_neighb


def create_mesh(x: ndarray, y: ndarray) -> Mesh:
    """Creates mesh from coordinates.

    Args:
        x, y: 1-D arrays with x and y coordinates.
    Returns:
        Mesh object.
    """

    mesh = Mesh()
    mesh_vertices = [[xi, yi] for xi, yi in zip(x, y)]
    mesh_cells = Triangulation(x, y).triangles

    editor = MeshEditor()
    editor.open(mesh, 'triangle', 2, 2)
    editor.init_vertices(len(mesh_vertices))
    editor.init_cells(len(mesh_cells))
    for vertex_index, v in enumerate(mesh_vertices):
        editor.add_vertex(vertex_index, v)
    for cell_index, c in enumerate(mesh_cells):
        editor.add_cell(cell_index, np.array(c, dtype=np.uintp))
    editor.close()

    return mesh


def compute_interp_data_linear(x: ndarray, x_neighb: ndarray, boundary_cond: str) -> tuple[coo_matrix, ndarray]:
    # Initialize interpoaltion matrix and BC vector.
    Phi = lil_matrix((len(x_neighb), len(x)), dtype=np.float32)
    b = np.zeros((len(x_neighb), 1), dtype=np.float32)

    # Create mesh and bounding box tree for fast computation of interpolation matrix.
    mesh = create_mesh(x[:, 0], x[:, 1])
    bbt = mesh.bounding_box_tree()
    bbt.build(mesh)

    # Compute interpoaltion matrix and BC vector.
    mesh_cells = list(cells(mesh))
    func_space = FunctionSpace(mesh, "CG", 1)
    d2v = dof_to_vertex_map(func_space)

    if boundary_cond == "periodic":
        x_neighb = apply_periodic_domain(x, x_neighb)

    for i, xi in enumerate(x_neighb):
        j = bbt.compute_first_entity_collision(Point(xi))
        if j < mesh.num_cells():
            for k, dof in enumerate(func_space.dofmap().cell_dofs(j)):
                value = func_space.element().evaluate_basis(k, xi, mesh_cells[j].get_vertex_coordinates(), mesh_cells[j].orientation())
                Phi[i, d2v[dof]] = value[0]
        else:
            b[i, 0] = OUT_OF_DOMAIN_NODE_VALUE

    return Phi.tocoo(), b


def compute_interp_data_knn(x: ndarray, x_neighb: ndarray, boundary_cond: str, k: int) -> tuple[coo_matrix, ndarray]:
    Phi = lil_matrix((len(x_neighb), len(x)), dtype=np.float32)
    b = np.zeros((len(x_neighb), 1), dtype=np.float32)

    hull = Delaunay(x)

    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(x)

    if boundary_cond == "periodic":
        x_neighb = apply_periodic_domain(x, x_neighb)

    for i, xi in enumerate(x_neighb):
        if hull.find_simplex(xi) == -1:
            b[i, 0] = OUT_OF_DOMAIN_NODE_VALUE
        else:
            _, indices = neigh.kneighbors(np.array([xi]))

            for idx in indices[0]:
                Phi[i, idx] = 1.0 / len(indices[0])

    return Phi.tocoo(), b


def compute_interp_data_idw(x: ndarray, x_neighb: ndarray, boundary_cond: str, k: int) -> tuple[coo_matrix, ndarray]:
    Phi = lil_matrix((len(x_neighb), len(x)), dtype=np.float32)
    b = np.zeros((len(x_neighb), 1), dtype=np.float32)

    hull = Delaunay(x)

    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(x)

    if boundary_cond == "periodic":
        x_neighb = apply_periodic_domain(x, x_neighb)

    for i, xi in enumerate(x_neighb):
        if hull.find_simplex(xi) == -1:
            b[i, 0] = OUT_OF_DOMAIN_NODE_VALUE
        else:
            distances, indices = neigh.kneighbors(np.array([xi]))
            weights = 1.0 / ((distances + 1e-5)**2)
            weights /= np.sum(weights)

            for idx, weight in zip(indices[0], weights[0]):
                Phi[i, idx] = weight

    return Phi.tocoo(), b


def compute_interp_data(x: ndarray, x_neighb: ndarray, interp: str, boundary_cond: str) -> tuple[coo_matrix, ndarray]:
    if interp == "linear":
        Phi, b = compute_interp_data_linear(x, x_neighb, boundary_cond)
    elif interp == "knn":
        Phi, b = compute_interp_data_knn(x, x_neighb, boundary_cond, k=5)
    elif interp == "idw":
        Phi, b = compute_interp_data_idw(x, x_neighb, boundary_cond, k=5)
    else:
        raise RuntimeError("Unsupported interpolation method")
    return Phi, b


def coord_to_interp_data(
    x: ndarray,
    interp: str,
    stencil_shape: str,
    stencil_size: float,
    boundary_cond: str
) -> tuple[coo_matrix, ndarray]:
    """Converts spatial coordinates to interpolation matrix and boundary condition vector.

    Args:
        x: Spatial coordinates. Has shape (N, 2).
        interp: Interpolation method.
        stencil_shape: Shape of the evaluation stencil (i.e., neighborhood shape).
        stencil_size: Size of the evaluation stencil.
        boundary_cond: Type of the boundary conditions.

    Returns:
        Phi: Interpolation matrix. Has shape (N_eval, N).
        b: Boundary condition vector. Has shape (N_eval, 1).
    """
    stencil = construct_stencil(stencil_shape, stencil_size)
    x_neighb = get_neighb_points(x, stencil)
    Phi, b = compute_interp_data(x, x_neighb, interp, boundary_cond)
    return Phi, b


class Interpolator(torch.nn.Module):
    def __call__(self, u: Tensor, Phi: Tensor, b: Tensor) -> Tensor:
        """Interpolates `u` given interpolatiuon matrix `Phi` and boundary condition vector `b`.

        Args:
            u: Values to be interpolated. Has shape (S, M, N, D_u).
            Phi: Interpolation matrices. Has shape (S, N_eval, N).
            b: Boundary condition vectors. Has shape (S, N_eval, 1).

        Returns:
            u_neighb: Interpolation of `u` evaluated at spatial neighborhood nodes.
                Has shape (S, M, N_eval, D_u).
        """
        S, M, _, D_u, N_eval = *u.shape, Phi.shape[1]
        u_neighb = torch.zeros((S, M, N_eval, D_u), dtype=u.dtype, device=u.device)
        for i in range(M):
            u_neighb[:, i] = torch.bmm(Phi, u[:, i]) + b  # add b to set out-of-domain node values to OUT_OF_DOMAIN_NODE_VALUE
        return u_neighb
