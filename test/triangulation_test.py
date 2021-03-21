from src.positioning.toa import toa_positioning
import matplotlib.pyplot as plt
import numpy as np

# test 1: LOS
toa_array = np.array([0.7530, 0.1623, 0.6255])*1e-6
zenith_array = np.array([1.7894, 2.6305, 1.7285])
bs_pos = np.array([[266.3800, 384.7510, -22.4043], [274.5290, 161.4730, -15.8938], [68.8520, 131.8180, -2.9075]])
_, pos3d_est = toa_positioning(bs_pos, toa_array, zenith_array)
print(f"The estimated 3D position of the UAV is: {pos3d_est}")

# test 2: LOS
toa_array = np.array([0.6968, 0.2038, 0.5993])*1e-6
zenith_array = np.array([1.8124, 2.3624, 1.7412])
_, pos3d_est = toa_positioning(bs_pos, toa_array, zenith_array)
print(f"The estimated 3D position of the UAV is: {pos3d_est}")

# test 3: NLOS with NLOS path corresponding to the one with lower delay
toa_array = np.array([7.3349e-07, 0.0340e-5, 0.0501e-5])
zenith_array = np.array([1.8195, 1.9859, 1.9353])
bs_pos = np.array([[274.5290, 161.4730, -15.8938], [68.8520, 131.8180, -2.9075], [126.4440, 355.1010, -15.2998]])
_, pos3d_est = toa_positioning(bs_pos, toa_array, zenith_array)
print(f"The estimated 3D position of the UAV is: {pos3d_est}")

# test 4: NLOS with NLOS path corresponding to the one with higher rx power
toa_array = np.array([0.1093, 0.0340, 0.0501])*1e-5
zenith_array = np.array([1.7368, 1.9859, 1.9353])
bs_pos = np.array([[274.5290, 161.4730, -15.8938], [68.8520, 131.8180, -2.9075], [126.4440, 355.1010, -15.2998]])
_, pos3d_est = toa_positioning(bs_pos, toa_array, zenith_array)
print(f"The estimated 3D position of the UAV is: {pos3d_est}")

radii = np.array([323.16863465, 93.27302902, 140.32826445])/1e3
figure, axes = plt.subplots()
draw_circle = plt.Circle((274.5290/1e3, 161.4730/1e3), radii[0], fill=False)
draw_circle2 = plt.Circle((68.8520/1e3, 131.8180/1e3), radii[1], fill=False)
draw_circle3 = plt.Circle((126.4440/1e3, 355.1010/1e3), radii[2], fill=False)
axes.set_aspect(1)
axes.add_artist(draw_circle)
axes.add_artist(draw_circle2)
axes.add_artist(draw_circle3)
plt.xlim(-1.25, 1.25)
plt.ylim(-1.25, 1.25)
plt.title('Triangulation')
plt.grid()
plt.show()

