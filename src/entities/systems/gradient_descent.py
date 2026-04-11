import math
import numpy as np

from entities.components.gd.optimizer_state import OptimizerAlgorithm, OptimizerState
from entities.components.gd.surface import GradientDescentSurface, LossFunctionType
from entities.components.transform import Transform
from entities.components.visuals.assets import AssetsState
from entities.components.visuals.visuals import Visuals
from entities.registry import Registry
from entities.systems.assets import AssetSystem
from math_utils import float1, vec3, create_transformation_matrix


class GradientDescentSurfaceSystem:
    @staticmethod
    def update(registry: Registry, time: float, delta_time: float):
        r_admin = registry.get_singleton(AssetsState)
        if r_admin is None:
            return
        admin_entity, (assets_state, ) = r_admin

        for g_entity, (g_transform, g_visuals, g_surface) in registry.view(Transform, Visuals, GradientDescentSurface):
            m_surf = create_transformation_matrix(g_transform.position, g_transform.rotation, g_transform.scale)
            children = registry.get_children(g_entity)

            try:
                m_inv = np.linalg.inv(m_surf)
            except np.linalg.LinAlgError:
                m_inv = np.identity(4, dtype=np.float32)

            if g_surface.dirty:
                GradientDescentSurfaceSystem._generate_mesh(assets_state, g_visuals, g_surface)
                g_surface.dirty = False

                for o_entity in children:
                    r_optimizer = registry.get_components(o_entity, Transform, OptimizerState)
                    if r_optimizer is None:
                        continue
                    o_transform, o_optimizer = r_optimizer

                    pos_world = np.array([o_transform.position[0], o_transform.position[1], o_transform.position[2], 1.0], dtype=np.float32)
                    pos_local = pos_world @ m_inv

                    new_local_y = float1(GradientDescentSurfaceSystem.evaluate_loss(
                        g_surface.function_type, pos_local[0], pos_local[2],
                        g_surface.rosenbrock_a, g_surface.rosenbrock_b
                    ))
                    pos_local[1] = new_local_y

                    new_pos_world = pos_local @ m_surf
                    o_transform.position = vec3(new_pos_world[0], new_pos_world[1], new_pos_world[2])

            do_step = False
            if g_surface.is_running:
                g_surface.step_timer += delta_time
                if g_surface.step_timer >= g_surface.step_interval:
                    do_step = True
                    g_surface.step_timer = 0.0

            for o_entity in children:
                r_optimizer = registry.get_components(o_entity, Transform, OptimizerState)
                if r_optimizer is None:
                    continue
                o_transform, o_optimizer = r_optimizer

                pos_world = np.array([o_transform.position[0], o_transform.position[1], o_transform.position[2], 1.0], dtype=np.float32)
                pos_local = pos_world @ m_inv

                local_x = pos_local[0]
                local_z = pos_local[2]

                if do_step:
                    dx, dz = GradientDescentSurfaceSystem.evaluate_gradient(
                        g_surface.function_type, local_x, local_z,
                        g_surface.rosenbrock_a, g_surface.rosenbrock_b
                    )

                    grad_mag = math.sqrt(dx**2 + dz**2)
                    max_grad = 1000.0
                    if grad_mag > max_grad:
                        dx = (dx / grad_mag) * max_grad
                        dz = (dz / grad_mag) * max_grad

                    if o_optimizer.algorithm == OptimizerAlgorithm.SGD:
                        dx += np.random.normal(0, o_optimizer.noise_scale)
                        dz += np.random.normal(0, o_optimizer.noise_scale)
                    elif o_optimizer.algorithm == OptimizerAlgorithm.MiniBatchSGD:
                        dx += np.random.normal(0, o_optimizer.noise_scale * 0.2)
                        dz += np.random.normal(0, o_optimizer.noise_scale * 0.2)

                    if o_optimizer.algorithm == OptimizerAlgorithm.Momentum:
                        o_optimizer.velocity_x = (o_optimizer.momentum_rate * o_optimizer.velocity_x) - (o_optimizer.learning_rate * dx)
                        o_optimizer.velocity_z = (o_optimizer.momentum_rate * o_optimizer.velocity_z) - (o_optimizer.learning_rate * dz)
                    else:
                        o_optimizer.velocity_x = -(o_optimizer.learning_rate * dx)
                        o_optimizer.velocity_z = -(o_optimizer.learning_rate * dz)

                    local_x += o_optimizer.velocity_x
                    local_z += o_optimizer.velocity_z

                local_y = float(GradientDescentSurfaceSystem.evaluate_loss(
                    g_surface.function_type, local_x, local_z,
                    g_surface.rosenbrock_a, g_surface.rosenbrock_b
                ))

                new_pos_local = np.array([local_x, local_y, local_z, 1.0], dtype=np.float32)
                new_pos_world = new_pos_local @ m_surf

                o_transform.position = vec3(new_pos_world[0], new_pos_world[1], new_pos_world[2])

                if do_step:
                    o_optimizer.trajectory.append(vec3(new_pos_world[0], new_pos_world[1], new_pos_world[2]))

            if do_step:
                g_surface.iterations += 1

    @staticmethod
    def evaluate_loss(func_type: LossFunctionType, x, z, a=1.0, b=100.0):
        if func_type == LossFunctionType.Himmelblau:
            return (x**2 + z - 11)**2 + (x + z**2 - 7)**2
        elif func_type == LossFunctionType.Rosenbrock:
            return (a - x)**2 + b * (z - x**2)**2
        elif func_type == LossFunctionType.Booth:
            return (x + 2*z - 7)**2 + (2*x + z - 5)**2
        return 0

    @staticmethod
    def evaluate_gradient(func_type: LossFunctionType, x, z, a=1.0, b=100.0):
        if func_type == LossFunctionType.Himmelblau:
            dx = 4*x*(x**2 + z - 11) + 2*(x + z**2 - 7)
            dz = 2*(x**2 + z - 11) + 4*z*(x + z**2 - 7)
        elif func_type == LossFunctionType.Rosenbrock:
            dx = -2*(a - x) - 4*b*x*(z - x**2)
            dz = 2*b*(z - x**2)
        elif func_type == LossFunctionType.Booth:
            dx = 2*(x + 2*z - 7) + 4*(2*x + z - 5)
            dz = 4*(x + 2*z - 7) + 2*(2*x + z - 5)
        else:
            dx, dz = 0, 0

        return dx, dz

    @staticmethod
    def _generate_mesh(assets_state: AssetsState, visuals: Visuals, surface: GradientDescentSurface):
        res = max(2, surface.resolution)
        s = surface.size / 2.0

        # == create initial grid ==
        x_lin = np.linspace(-s, s, res, dtype=np.float32)
        z_lin = np.linspace(-s, s, res, dtype=np.float32)
        X, Z = np.meshgrid(x_lin, z_lin)

        # == evaluate the expression (vectorized) ==
        Y = GradientDescentSurfaceSystem.evaluate_loss(
            surface.function_type, X, Z,
            surface.rosenbrock_a, surface.rosenbrock_b
        ).astype(np.float32)

        # == calculate analytical normals ==
        dX, dZ = GradientDescentSurfaceSystem.evaluate_gradient(
            surface.function_type, X, Z,
            surface.rosenbrock_a, surface.rosenbrock_b
        )
        # surface normal formula: (-df/dx, 1, -df/dz)
        Nx = -dX.astype(np.float32) # type: ignore
        Ny = np.ones_like(Y, dtype=np.float32)
        Nz = -dZ.astype(np.float32) # type: ignore

        # == stack and normalize ==
        normals = np.stack([Nx, Ny, Nz], axis=-1)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

        # == calculate UVs ==
        U, V = np.meshgrid(np.linspace(0, 1, res, dtype=np.float32), np.linspace(0, 1, res, dtype=np.float32))

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

        tris1 = np.stack([i1, i3, i2], axis=-1)
        tris2 = np.stack([i2, i3, i4], axis=-1)
        indices_flat = np.concatenate([tris1, tris2], axis=0).flatten().astype(np.uint32)

        visuals.mesh = AssetSystem.create_immediate_mesh(assets_state, vertices_flat, indices_flat)
