import math
import numpy as np


def _create_subdivided_sphere(base_verts, base_faces, radius, subdivisions):
    vertices = [[v[0], v[1], v[2]] for v in base_verts]
    faces = [list(f) for f in base_faces]

    for i in range(len(vertices)):
        l = math.sqrt(sum(x * x for x in vertices[i]))
        vertices[i] = [x * radius / l for x in vertices[i]]

    for _ in range(subdivisions):
        cache = {}
        new_faces = []

        def get_mid(i1, i2):
            pair = tuple(sorted((i1, i2)))
            if pair in cache:
                return cache[pair]
            v1, v2 = vertices[i1], vertices[i2]
            mx, my, mz = v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]
            l = math.sqrt(mx * mx + my * my + mz * mz)
            vertices.append([mx * radius / l, my * radius / l, mz * radius / l])
            idx = len(vertices) - 1
            cache[pair] = idx
            return idx

        for face in faces:
            v0, v1, v2 = face
            m01 = get_mid(v0, v1)
            m12 = get_mid(v1, v2)
            m20 = get_mid(v2, v0)
            new_faces.extend([
                [v0, m01, m20], [v1, m12, m01],
                [v2, m20, m12], [m01, m12, m20]
            ])
        faces = new_faces

    v_data = []
    for v in vertices:
        x, y, z = v
        nx, ny, nz = x / radius, y / radius, z / radius
        clamped_ny = max(-1.0, min(1.0, ny))
        u = 0.5 + math.atan2(z, x) / (2 * math.pi)
        v_tex = 0.5 - math.asin(clamped_ny) / math.pi
        v_data.append([x, y, z, nx, ny, nz, u, v_tex])

    final_vertices = list(v_data)
    final_indices = []

    for face in faces:
        i0, i1, i2 = face
        u0, u1, u2 = final_vertices[i0][6], final_vertices[i1][6], final_vertices[i2][6]

        if max(u0, u1, u2) - min(u0, u1, u2) > 0.5:
            face_idx = [i0, i1, i2]
            for j in range(3):
                v_idx = face_idx[j]
                if final_vertices[v_idx][6] < 0.25:
                    new_v = list(final_vertices[v_idx])
                    new_v[6] += 1.0  # Shift U up by 1 to maintain continuity
                    final_vertices.append(new_v)
                    face_idx[j] = len(final_vertices) - 1
            final_indices.extend(face_idx)
        else:
            final_indices.extend([i0, i1, i2])

    for i in range(0, len(final_indices), 3):
        face_idx = [final_indices[i], final_indices[i+1], final_indices[i+2]]
        for j in range(3):
            v_idx = face_idx[j]
            y_val = final_vertices[v_idx][1] / radius
            if abs(y_val) > 0.9999:
                other1 = final_vertices[face_idx[(j+1) % 3]]
                other2 = final_vertices[face_idx[(j+2) % 3]]
                avg_u = (other1[6] + other2[6]) / 2.0
                new_v = list(final_vertices[v_idx])
                new_v[6] = avg_u
                final_vertices.append(new_v)
                final_indices[i+j] = len(final_vertices) - 1

    flat_data = []
    for v in final_vertices:
        flat_data.extend(v)

    return np.array(flat_data, dtype=np.float32), np.array(final_indices, dtype=np.uint32)


def generate_tetrasphere(radius=1.0, subdivisions=3):
    """
    Generates a tetrasphere.

    Returns:
        vertices: np.array (float32) -> Interleaved [Pos(3), Norm(3), UV(2)]
        indices: np.array (uint32) -> EBO indices
    """
    base_verts = [
        [1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]
    ]
    base_faces = [
        [0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]
    ]
    return _create_subdivided_sphere(base_verts, base_faces, radius, subdivisions)


def generate_icosphere(radius=1.0, subdivisions=3):
    """
    Generates an icosphere.

    Returns:
        vertices: np.array (float32) -> Interleaved [Pos(3), Norm(3), UV(2)]
        indices: np.array (uint32) -> EBO indices
    """
    t = (1.0 + math.sqrt(5.0)) / 2.0
    base_verts = [
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ]
    base_faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    return _create_subdivided_sphere(base_verts, base_faces, radius, subdivisions)