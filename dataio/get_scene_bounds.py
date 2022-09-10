import numpy as np


def get_scene_bounds(scene_name):
    if scene_name == 'scene0000_00':
        x_min, x_max = -0.2, 8.6
        y_min, y_max = -0.2, 8.9
        z_min, z_max = -0.2, 3.2

    elif scene_name == 'scene0002_00':
        x_min, x_max = 0.6, 5.0
        y_min, y_max = -0.2, 5.8
        z_min, z_max = 0.0, 3.5

    elif scene_name == 'scene0005_00':
        x_min, x_max = -0.24, 5.55
        y_min, y_max = 0.30, 5.55
        z_min, z_max = -0.24, 2.65

    elif scene_name == 'scene0006_00':
        x_min, x_max = -0.08, 4.13
        y_min, y_max = -0.18, 7.40
        z_min, z_max = -0.06, 2.65

    elif scene_name == 'scene0012_00':
        x_min, x_max = -0.20, 5.60
        y_min, y_max = -0.20, 5.50
        z_min, z_max = -0.20, 2.70

    elif scene_name == 'scene0024_00':
        x_min, x_max = -0.20, 7.38
        y_min, y_max = -0.20, 8.19
        z_min, z_max = -0.20, 2.65

    elif scene_name == 'scene0050_00':
        x_min, x_max = 0.80, 6.70
        y_min, y_max = 0.10, 4.60
        z_min, z_max = -0.20, 2.90

    elif scene_name == 'scene0054_00':
        x_min, x_max = -1.4, 1.4
        y_min, y_max = -0.3, 1.4
        z_min, z_max = -1.4, 1.4

    elif scene_name == 'whiteroom':
        x_min, x_max = -2.46, 3.06
        y_min, y_max = -0.3, 3.5
        z_min, z_max = 0.36, 8.2

    elif scene_name == 'kitchen':
        x_min, x_max = -3.20, 3.80
        y_min, y_max = -0.2, 3.20
        z_min, z_max = -3.20, 5.50

    elif scene_name == 'breakfast_room':
        x_min, x_max = -2.23, 1.85
        y_min, y_max = -0.5, 2.77
        z_min, z_max = -1.7, 3.0

    elif scene_name == 'staircase':
        x_min, x_max = -4.20, 2.60
        y_min, y_max = -0.2, 3.5
        z_min, z_max = -5.3, 1.2

    elif scene_name == 'icl_living_room':
        x_min, x_max = -2.6, 2.7
        y_min, y_max = -0.1, 2.8
        z_min, z_max = -2.2, 3.2

    elif scene_name == 'complete_kitchen':
        x_min, x_max = -5.60, 3.70
        y_min, y_max = -0.1, 3.2
        z_min, z_max = -6.50, 3.50

    elif scene_name == 'green_room':
        x_min, x_max = -2.5, 5.5
        y_min, y_max = -0.2, 2.9
        z_min, z_max = 0.3, 5.0

    elif scene_name == 'grey_white_room':
        x_min, x_max = -0.55, 5.3
        y_min, y_max = -0.1, 3.0
        z_min, z_max = -3.75, 0.65

    elif scene_name == 'morning_apartment':
        x_min, x_max = -1.38, 2.10
        y_min, y_max = -0.20, 2.10
        z_min, z_max = -2.20, 1.75

    elif scene_name == 'thin_geometry':
        x_min, x_max = -2.35, 1.00
        y_min, y_max = -0.20, 1.00
        z_min, z_max = 0.20, 3.80

    else:
        raise NotImplementedError

    return np.array([[x_min, x_max], [y_min, y_max], [z_min, z_max]])

