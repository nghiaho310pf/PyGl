import numpy as np
import math
from entities.components.surface_function import SurfaceFunction, CompilationStatus
from entities.components.transform import Transform
from entities.components.visuals import Visuals
from entities.registry import Registry
from meshes.mesh import Mesh


class FunctionSurfaceSystem:
    @staticmethod
    def update(registry: Registry, time: float, delta_time: float):
        for entity, (transform, visuals, fun) in registry.view(Transform, Visuals, SurfaceFunction):
            if not fun.generated and not fun.expression_dirty:
                FunctionSurfaceSystem._generate_mesh(visuals, fun)
                fun.generated = True

    @staticmethod
    def _generate_mesh(visuals: Visuals, fun: SurfaceFunction):
        res = max(2, fun.resolution)
        s = fun.size / 2.0

        # == create initial grid ==
        x_lin = np.linspace(-s, s, res, dtype=np.float32)
        z_lin = np.linspace(-s, s, res, dtype=np.float32)
        X, Z = np.meshgrid(x_lin, z_lin)

        # == evaluate the expression ==
        safe_env = {"np": np, "math": math}
        local_env = {"x": X, "z": Z}
        try:
            Y = eval(fun.expression, safe_env, local_env)
            # if the expression was a constant (like "0"), broadcast it to the grid shape
            Y = np.broadcast_to(Y, X.shape).astype(np.float32)

            fun.error_status = CompilationStatus.Ok
            fun.error_string = "OK"
        except Exception as e:
            fun.error_status = CompilationStatus.Error
            fun.error_string = f"{type(e).__name__}: {str(e)}"
            return

        # == calculate normals ==
        dx = fun.size / (res - 1)
        dz = fun.size / (res - 1)

        dY_dZ, dY_dX = np.gradient(Y, dz, dx)

        Nx = -dY_dX
        Ny = np.ones_like(Y)
        Nz = -dY_dZ

        # == stack and normalize ==
        normals = np.stack([Nx, Ny, Nz], axis=-1)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

        # == calculate UVs ==
        U, V = np.meshgrid(np.linspace(0, 1, res), np.linspace(0, 1, res))

        # == build vertex array ==
        vertices = np.empty((res, res, 8), dtype=np.float32)
        vertices[..., 0] = X
        vertices[..., 1] = Y
        vertices[..., 2] = Z
        vertices[..., 3:6] = normals
        vertices[..., 6] = U
        vertices[..., 7] = V
        vertices_flat = vertices.reshape(-1)

        # == build indices ==
        r = np.arange(res - 1)
        c = np.arange(res - 1)
        rr, cc = np.meshgrid(r, c, indexing='ij')

        i1 = rr * res + cc
        i2 = rr * res + (cc + 1)
        i3 = (rr + 1) * res + cc
        i4 = (rr + 1) * res + (cc + 1)

        # two triangles per grid square
        tris1 = np.stack([i1, i3, i2], axis=-1)
        tris2 = np.stack([i2, i3, i4], axis=-1)
        indices_flat = np.concatenate(
            [tris1, tris2], axis=0).flatten().astype(np.uint32)

        visuals.mesh = Mesh(vertices_flat, indices_flat)
