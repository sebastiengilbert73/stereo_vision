import numpy as np

class ProjectionMatrix:
    def __init__(self, xy_XYZ_tuples=None):
        self.matrix = np.eye(3, 4, dtype=float)
        if xy_XYZ_tuples is not None:
            self.Create(xy_XYZ_tuples)

    def Create(self, xy_XYZ_tuples):
        if len(xy_XYZ_tuples) < 6:
            raise ValueError(f"ProjectionMatrix.Create(): len(xy_XYZ_tuples) ({len(xy_XYZ_tuples)}) < 6")
        """
          p00 ...                    ...  p23    1/lambda_i
        | ...
        | Xi Yi Zi 1  0  0  0  0  0  0  0  0  0 ... -xi 0 ... |   | p00 |
        | 0  0  0  0  Xi Yi Zi 1  0  0  0  0  0 ... -yi 0 ... |   | p01 |
        | 0  0  0  0  0  0  0  0  Xi Yi Zi 1  0 ... -1  0 ... |   | p02 |
                                                                  | ... |
                                                                  | 1/lambda0 |
                                                                  | 1/lambda1 |
                                                                  | ... |
                                                                  | 1/lambda_N-1 |
        """
        A = np.zeros((3 * len(xy_XYZ_tuples), 12 + len(xy_XYZ_tuples)), dtype=float)
        for corr_ndx in range(len(xy_XYZ_tuples)):
            xy = xy_XYZ_tuples[corr_ndx][0]
            XYZ = xy_XYZ_tuples[corr_ndx][1]
            A[3 * corr_ndx, 0] = XYZ[0]
            A[3 * corr_ndx, 1] = XYZ[1]
            A[3 * corr_ndx, 2] = XYZ[2]
            A[3 * corr_ndx, 3] = 1.0
            A[3 * corr_ndx, 12 + corr_ndx] = -xy[0]
            A[3 * corr_ndx + 1, 4] = XYZ[0]
            A[3 * corr_ndx + 1, 5] = XYZ[1]
            A[3 * corr_ndx + 1, 6] = XYZ[2]
            A[3 * corr_ndx + 1, 7] = 1.0
            A[3 * corr_ndx + 1, 12 + corr_ndx] = -xy[1]
            A[3 * corr_ndx + 2, 8] = XYZ[0]
            A[3 * corr_ndx + 2, 9] = XYZ[1]
            A[3 * corr_ndx + 2, 10] = XYZ[2]
            A[3 * corr_ndx + 2, 11] = 1.0
            A[3 * corr_ndx + 2, 12 + corr_ndx] = -1.0
        # Solve homogeneous system of linear equations
        # Cf. https://stackoverflow.com/questions/1835246/how-to-solve-homogeneous-linear-equations-with-numpy
        # Find the eigenvalues and eigenvector of A^T A
        e_vals, e_vecs = np.linalg.eig(np.dot(A.T, A))
        # Extract the eigenvector (column) associated with the minimum eigenvalue
        z = e_vecs[:, np.argmin(e_vals)]
        # Since the coefficients are defined up to a scale factor (we solved a homogeneous system of linear equations), we can multiply them by an arbitrary constant
        z = z/z[11]

        self.matrix[0, 0] = z[0]
        self.matrix[0, 1] = z[1]
        self.matrix[0, 2] = z[2]
        self.matrix[0, 3] = z[3]
        self.matrix[1, 0] = z[4]
        self.matrix[1, 1] = z[5]
        self.matrix[1, 2] = z[6]
        self.matrix[1, 3] = z[7]
        self.matrix[2, 0] = z[8]
        self.matrix[2, 1] = z[9]
        self.matrix[2, 2] = z[10]
        self.matrix[2, 3] = z[11]

    def Project(self, point3D, must_round=False, zero_threshold=1e-9):
        if len(point3D) != 3:
            raise ValueError(f"ProjectionMatrix.Project(): len(point3D) ({len(point3D)}) != 3")
        xyz1 = np.ones(4)
        xyz1[0] = point3D[0]
        xyz1[1] = point3D[1]
        xyz1[2] = point3D[2]
        projection = self.matrix @ xyz1
        #print (f"Project(): projection = {projection}")
        if abs(projection[2]) < zero_threshold:
            raise ZeroDivisionError(f"abs(projection[2]) ({abs(projection[2])}) < zero_threshold ({zero_threshold})")
        xy = [projection[0]/projection[2], projection[1]/projection[2]]
        if must_round:
            xy[0] = round(xy[0])
            xy[1] = round(xy[1])
        return xy
