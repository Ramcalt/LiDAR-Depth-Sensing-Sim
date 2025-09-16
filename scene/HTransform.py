import numpy as np

class HTransform:
    """Homogenous Transform Class"""
    matrix: np.ndarray

    def __init__(self, matrix=None):
        """Initialise 4x4 homogenous transform from supplied matrix, or otherwise generate an identity matrix."""
        if matrix is None:
            self.matrix = np.eye(4)
        else:
            if matrix.shape != (4, 4):
                raise ValueError("Transform matrix must be 4x4.")
            self.matrix = matrix.asType(float)

    def __matmul__(self, other):
        """Overload for HTransform."""
        if isinstance(other, HTransform):
            return HTransform(self.matrix @ other.matrix)
        elif isinstance(other, np.ndarray):
            if other.shape != (4,):
                raise ValueError("Vector size must be 4.")
            return HTransform(self.matrix @ other)
        else:
            raise TypeError("Unsupported operand type for @")

    @staticmethod
    def translation(tx, ty, tz):
        """Create a translation HTransform."""
        T = np.eye(4)
        T[:3, 3] = [tx, ty, tz]
        return HTransform(T)

    @staticmethod
    def scaling(sx, sy, sz):
        """Create a scaling HTransform."""
        S = np.diag([sx, sy, sz, 1])
        return HTransform(S)

    # @staticmethod
    # def rotation(axis, theta):
    #     """Create a rotation HTransform along axis u."""
    #     if isinstance(axis, np.ndarray) and axis.shape == (4, ):
    #         r = axis @ np.linalg.norm(axis)
    #         s = r
    #         i = np.argmin(r[0:2])
    #         s[i] = 0
    #         temp = s[(i + 1) % 3]
    #         s[(i + 1) % 3] = s[(i - 1) % 3]
    #         s[(i - 1) % 3] = temp
    #         t = np.cross(r, s)
    #         M = np.array(r, s, t)
    #         c = np.cos(theta)
    #         s = np.sin(theta)
    #         Rx = np.array([
    #             [1, 0, 0, 0],
    #             [0, c, -s, 0],
    #             [0, s, c, 0],
    #             [0, 0, 0, 1]
    #         ])
    #         MT = np.transpose(M)
    #         return HTransform(M @ Rx @ MT)
    #     else:
    #         raise ValueError("Axis vector must be a ndarray of length 4.")

    @staticmethod
    def rotation_x(theta):
        """Rotation around X-axis (radians)."""
        c, s = np.cos(theta), np.sin(theta)
        m = np.array([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1]
        ])
        return HTransform(m)

    @staticmethod
    def rotation_y(theta):
        """Rotation around Y-axis (radians)."""
        c, s = np.cos(theta), np.sin(theta)
        m = np.array([
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1]
        ])
        return HTransform(m)

    @staticmethod
    def rotation_z(theta):
        """Rotation around Z-axis (radians)."""
        c, s = np.cos(theta), np.sin(theta)
        m = np.array([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        return HTransform(m)