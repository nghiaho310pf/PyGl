import numpy as np

import math_utils


class Entity:
    def __init__(self, mesh, position=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1)):
        self.mesh = mesh
        self.position = np.array(position, dtype=np.float32)
        self.rotation = np.array(rotation, dtype=np.float32)
        self.scale = np.array(scale, dtype=np.float32)

    def get_model_matrix(self):
        return math_utils.create_transformation_matrix(self.position, self.rotation, self.scale)

    def update(self, dt):
        pass
