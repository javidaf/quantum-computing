import numpy as np

import matplotlib.pyplot as plt


def main():

    ket0 = np.array([1, 0])
    ket1 = np.array([0, 1])

    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])

    sigma_x_ket0 = sigma_x @ ket0
    sigma_y_ket0 = sigma_y @ ket0
    sigma_z_ket0 = sigma_z @ ket0

    print(sigma_x_ket0)
    print(sigma_y_ket0)
    print(sigma_z_ket0)

    sigma_x_ket1 = np.dot(sigma_x, ket1)
    sigma_y_ket1 = np.dot(sigma_y, ket1)
    sigma_z_ket1 = np.dot(sigma_z, ket1)
    print("sigma_x|1>:", sigma_x_ket1)
    print("sigma_y|1>:", sigma_y_ket1)
    print("sigma_z|1>:", sigma_z_ket1)
if __name__ == "__main__":
    main()
