"""
© 2021, New York University, Tandon School of Engineering, NYU WIRELESS.
"""
import numpy as np
from scipy.constants import speed_of_light
EPSILON = 10


def calculate_three_circle_intersection(bss_centers, radius):

    x0, y0 = tuple(bss_centers[0])
    x1, y1 = tuple(bss_centers[1])
    x2, y2 = tuple(bss_centers[2])
    r0, r1, r2 = tuple(radius)
    # pick the first two circle and get the distance between them
    dx = x1-x0
    dy = y1-y0
    d12 = np.sqrt(dx**2 + dy**2)
    if d12 > (r0+r1):
        # the first two circles do not intersect
        return False, "circles do not intersect"
    if d12 < np.abs(r0-r1):
        return False, "one circle is contained in the other one"
    a = (r0**2 - r1**2 + d12**2) / (2.0 * d12)
    # intersection point between the line connecting the intersection points and the line connecting the centers of
    # the circles
    point_x = x0 + (dx*a/d12)
    point_y = y0 + (dy*a/d12)
    h = np.sqrt(r0**2 - a**2)
    rx = -dy*(h/d12)
    ry = dx*(h/d12)
    # determine the two intersection points
    int_1_x = point_x + rx
    int_1_y = point_y + ry
    int_2_x = point_x - rx
    int_2_y = point_y - ry

    # now determine if circle 3 intersects any of the two intersection points
    d1 = np.sqrt((int_1_x-x2)**2 + (int_1_y-y2)**2)
    d2 = np.sqrt((int_2_x-x2)**2 + (int_2_y-y2)**2)

    if np.abs(d1-r2) < EPSILON:
        return True, [int_1_x, int_1_y]
    elif np.abs(d2-r2) < EPSILON:
        return True, [int_2_x, int_2_y]
    else:
        return False, "intersection not found"


def toa_positioning(bss_pos, tof, zoa):

    """Computes the UAV position using the ToA (Time of Arrival) technique

        Parameters
        ----------
        bss_pos : np.array
            Array with the positions of the Base Stations in this format: [x,y,z] where z is the height of the BS

        tof : np.array
            Array containing the ToF (Time of Flight) from each BS to the UAV
            The ToF is time it took for the signal to travel from BS_i to the UAV to be located

        zoa: np.array
            Array with the zenith (elevation) of arrival [rads] of the path from BS_i to the UAV

        Returns
        -------
        array
            an array with the estimated coordinates of the UAV as follows: [x,y,z] where z is the estimated height
    """
    bss_pos, tof, zoa = np.array(bss_pos), np.array(tof), np.array(zoa)
    assert bss_pos.shape[0] == 3, "At least 3 BSs needed to triangulate the UAV"
    assert bss_pos.shape[0] == tof.shape[0], "# of BSs and # of ToF measurements must be the same"

    # compute the 3D distance traveled by signals from each BS to the UAV
    d = speed_of_light*tof
    # estimated height of the UAV from each BS
    height = -np.cos(zoa)*d
    est_height = np.mean(height + bss_pos[:, -1])
    # estimated 2D distance traveled by the signals from each BS to the UAV
    est_2d_distance = np.sin(zoa)*d
    est_2d_exist, est_2d_pos = calculate_three_circle_intersection(bss_pos[:, :2], est_2d_distance)

    if not est_2d_exist:
        return False, "Impossible to determine the 2D position of the UAV: "+est_2d_pos
    else:
        return True, [est_2d_pos[0], est_2d_pos[1], est_height]
