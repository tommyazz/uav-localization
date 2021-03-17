from src.positioning.toa import calculate_three_circle_intersection
import numpy as np

bs_pos = np.array([[-2, 0, 10], [1, 0, 10], [0, 4, 10]])
rad = np.array([2, 1, 4])
pos2d_exist, pos2d_est = calculate_three_circle_intersection(bs_pos[:, :2], rad)

if pos2d_exist:
    print(f"The estimated UAV position is: {pos2d_est}")
