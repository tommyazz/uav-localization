from src.positioning.toa import calculate_three_circle_intersection, toa_positioning
import numpy as np

# test 1: LOS
toa_array = np.array([0.7530, 0.1623, 0.6255])*1e-6
zenith_array = np.array([1.7894, 2.6305, 1.7285])
bs_pos = np.array([[266.3800, 384.7510, -22.4043], [274.5290, 161.4730, -15.8938], [68.8520, 131.8180, -2.9075]])
pos3d_est = toa_positioning(bs_pos, toa_array, zenith_array)
print(f"The estimated 3D position of the UAV is: {pos3d_est}")

# test 2: LOS
toa_array = np.array([0.6968, 0.2038, 0.5993])*1e-6
zenith_array = np.array([1.8124, 2.3624, 1.7412])
pos3d_est = toa_positioning(bs_pos, toa_array, zenith_array)
print(f"The estimated 3D position of the UAV is: {pos3d_est}")

# test 3: NLOS
toa_array = np.array([0.1093, 0.0340, 0.0501])*1e-5
zenith_array = np.array([1.7368, 1.9859, 1.9353])
bs_pos = np.array([[274.5290, 161.4730, -15.8938], [68.8520, 131.8180, -2.9075], [126.4440, 355.1010, -15.2998]])
pos3d_est = toa_positioning(bs_pos, toa_array, zenith_array)
print(f"The estimated 3D position of the UAV is: {pos3d_est}")
