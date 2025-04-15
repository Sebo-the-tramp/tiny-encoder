import numpy as np

def quaternion_to_euler(x, y, z, w):
    # Roll (x-axis rotation)
    sinr = 2 * (w * x + y * z)
    cosr = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr, cosr)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))  # Clamp to avoid invalid domain

    # Yaw (z-axis rotation)
    siny = 2 * (w * z + x * y)
    cosy = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny, cosy)

    return roll, pitch, yaw