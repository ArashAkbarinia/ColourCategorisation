import numpy as np
import cv2
import scipy.io
import math

def rgb2belonging(image_rgb, configs_mat):
    """
    Labels each pixel in the image as one of the focal eleven colors.
    
    Parameters:
    -----------
    image_rgb : numpy.ndarray
        RGB image to process
    configs_mat : str
        Path to MATLAB .mat file containing the color ellipsoids configurations
        
    Returns:
    --------
    belonging_image : numpy.ndarray
        Image where each pixel belongs to one of the ellipsoids
    """
    # Check if image is normalized between 0 and 1
    if np.max(image_rgb) <= 1:
        image_rgb = (image_rgb * 255).astype(np.uint8)
    
    # Load configurations
    configs = np.load(configs_mat)
    color_ellipsoids = configs['ColourEllipsoids']
    
    # Convert RGB to LAB
    image_opponent = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(float)
    
    # Calculate belongings
    belonging_image = all_ellipsoids_evaluate_belonging(image_opponent, color_ellipsoids)
    
    return belonging_image

def all_ellipsoids_evaluate_belonging(input_image, color_ellipsoids):
    """
    Computes belonging of each pixel for all ellipsoids.
    
    Parameters:
    -----------
    input_image : numpy.ndarray
        The input image
    color_ellipsoids : numpy.ndarray
        The color ellipsoids parameters
        
    Returns:
    --------
    belongings : numpy.ndarray
        The belonging matrix for each pixel to each ellipsoid
    """
    n_ellipsoids = color_ellipsoids.shape[0]
    
    if len(input_image.shape) == 3:
        rows, cols, chns = input_image.shape
        # Reshape the image into a giant vector where every row corresponds to a pixel
        input_image_reshaped = input_image.reshape(rows * cols, chns)
        vector_rows = rows * cols
    else:
        vector_rows = input_image.shape[0]
        input_image_reshaped = input_image
    
    belongings = np.zeros((vector_rows, n_ellipsoids))
    
    for i in range(n_ellipsoids):
        ibelonging, _ = ellipsoid_evaluate_belonging(input_image_reshaped, color_ellipsoids[i, :])
        belongings[:, i] = ibelonging
    
    if len(input_image.shape) == 3:
        belongings = belongings.reshape(rows, cols, n_ellipsoids)
    
    return belongings

def ellipsoid_evaluate_belonging(points, ellipsoid):
    """
    Computes the belonging of each point to the given ellipsoid.
    
    Parameters:
    -----------
    points : numpy.ndarray
        The input points
    ellipsoid : numpy.ndarray
        The parameters of the ellipsoid
        
    Returns:
    --------
    belonging : numpy.ndarray
        The belonging value of each point to the ellipsoid
    distances : numpy.ndarray
        The distance of each point to the ellipsoid
    """
    if len(points.shape) == 3:
        rows, cols, chns = points.shape
        points_reshaped = points.reshape(rows * cols, chns)
    else:
        points_reshaped = points
        rows, cols = points.shape[0], 1
    
    centre_x = ellipsoid[0]
    centre_y = ellipsoid[1]
    centre_z = ellipsoid[2]
    
    distances, intersection = distance_ellipsoid(points_reshaped, ellipsoid)
    
    # Distances from the centre to the closest points in the ellipse
    h = np.sqrt((intersection[:, 0] - centre_x) ** 2 + 
                (intersection[:, 1] - centre_y) ** 2 + 
                (intersection[:, 2] - centre_z) ** 2)
    
    # Distances from the centre to the points
    x = np.sqrt((points_reshaped[:, 0] - centre_x) ** 2 + 
                (points_reshaped[:, 1] - centre_y) ** 2 + 
                (points_reshaped[:, 2] - centre_z) ** 2)
    
    g = ellipsoid[9]  # Using index 9 as in MATLAB (10th element, 0-indexed in Python)
    
    belonging = 1 / (1 + np.exp(g * (x - h)))
    
    if len(points.shape) == 3:
        belonging = belonging.reshape(rows, cols, 1)
    
    return belonging, distances

def distance_ellipsoid(points, ellipsoid):
    """
    Calculate distance from points to ellipsoid.
    
    Parameters:
    -----------
    points : numpy.ndarray
        The polar coordinates of the points [x, y, z]
    ellipsoid : numpy.ndarray
        Ellipsoid parameters: [cx, cy, cz, ax, ay, az, rx, ry, rz, ...]
        
    Returns:
    --------
    distance : numpy.ndarray
        The distance from the point to the ellipsoid
    intersection : numpy.ndarray
        The intersection points on ellipsoid surface
    """
    cx = ellipsoid[0]
    cy = ellipsoid[1]
    cz = ellipsoid[2]
    ax = ellipsoid[3]
    ay = ellipsoid[4] + 1e-10  # To avoid division by 0
    az = ellipsoid[5] + 1e-10  # To avoid division by 0
    rx = ellipsoid[6]
    ry = ellipsoid[7]
    rz = ellipsoid[8]
    
    rows = points.shape[0]
    
    # Centre the points relatively to the ellipsoid
    transferred_point = points - np.tile([cx, cy, cz], (rows, 1))
    
    # Rotate all points an angle alpha so that we can reduce the problem to one of canonical ellipsoids
    rotx = create_rotation_x(rx)
    roty = create_rotation_y(ry)
    rotz = create_rotation_z(rz)
    
    # Matrix multiplication in numpy for 3D transformations
    rot = np.matmul(np.matmul(rotz, roty), rotx)
    transferred_point = transform_point3(transferred_point, rot)
    
    px = transferred_point[:, 0] + 1e-10  # To avoid division by 0
    py = transferred_point[:, 1]
    pz = transferred_point[:, 2]
    
    # Calculate the intersection points on the surface
    x1 = 1 / np.sqrt(1 / (ax ** 2) + (py / px / ay) ** 2 + (pz / px / az) ** 2)
    x2 = -x1
    y1 = py / px * x1
    y2 = py / px * x2
    z1 = pz / px * x1
    z2 = pz / px * x2
    
    # Calculating the distance to the point
    d1 = np.sqrt((px - x1) ** 2 + (py - y1) ** 2 + (pz - z1) ** 2)
    d2 = np.sqrt((px - x2) ** 2 + (py - y2) ** 2 + (pz - z2) ** 2)
    distance = np.minimum(d1, d2)
    
    # Closest points in the ellipse
    d1_less_or_equal_d2 = (d1 <= d2)
    d1_greater_or_equal_d2 = (d1 >= d2)
    
    x_intersection = x1 * d1_less_or_equal_d2 + x2 * d1_greater_or_equal_d2
    y_intersection = y1 * d1_less_or_equal_d2 + y2 * d1_greater_or_equal_d2
    z_intersection = z1 * d1_less_or_equal_d2 + z2 * d1_greater_or_equal_d2
    
    intersection = np.column_stack((x_intersection, y_intersection, z_intersection))
    
    # Transferring it back with the rotation and translation
    intersection = transform_point3(intersection, rot.T)
    intersection = intersection + np.tile([cx, cy, cz], (rows, 1))
    
    return distance, intersection

def create_rotation_x(alpha):
    """
    Create rotation matrix for the X direction.
    
    Parameters:
    -----------
    alpha : float
        The rotation angle
        
    Returns:
    --------
    rotation_mat : numpy.ndarray
        The rotation matrix 4x4
    """
    s = math.sin(alpha)
    c = math.cos(alpha)
    
    rotation_mat = np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ])
    
    return rotation_mat

def create_rotation_y(alpha):
    """
    Create rotation matrix for the Y direction.
    
    Parameters:
    -----------
    alpha : float
        The rotation angle
        
    Returns:
    --------
    rotation_mat : numpy.ndarray
        The rotation matrix 4x4
    """
    s = math.sin(alpha)
    c = math.cos(alpha)
    
    rotation_mat = np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ])
    
    return rotation_mat

def create_rotation_z(alpha):
    """
    Create rotation matrix for the Z direction.
    
    Parameters:
    -----------
    alpha : float
        The rotation angle
        
    Returns:
    --------
    rotation_mat : numpy.ndarray
        The rotation matrix 4x4
    """
    s = math.sin(alpha)
    c = math.cos(alpha)
    
    rotation_mat = np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    return rotation_mat

def transform_point3(point, trans):
    """
    Transform point with the given transformation.
    
    Parameters:
    -----------
    point : numpy.ndarray
        The point in 3D
    trans : numpy.ndarray
        The transformation matrix
        
    Returns:
    --------
    transformed_point : numpy.ndarray
        The transformed point
    """
    if len(point.shape) == 3:
        rows, cols, chns = point.shape
        x = point[:, :, 0].flatten()
        y = point[:, :, 1].flatten()
        z = point[:, :, 2].flatten()
        np_points = rows * cols
    elif point.shape[1] == 3:
        x = point[:, 0]
        y = point[:, 1]
        z = point[:, 2]
        np_points = point.shape[0]
    else:
        x = point[0, :]
        y = point[1, :]
        z = point[2, :]
        np_points = point.shape[1]
    
    # Create homogeneous coordinates
    homogeneous_points = np.column_stack((x, y, z, np.ones(np_points, dtype=point.dtype)))
    
    # Apply transformation
    transformed_homogeneous = np.matmul(homogeneous_points, trans)
    
    # Convert back from homogeneous coordinates
    transformed_point = transformed_homogeneous[:, :3] / np.column_stack((
        transformed_homogeneous[:, 3], 
        transformed_homogeneous[:, 3], 
        transformed_homogeneous[:, 3]
    ))
    
    if len(point.shape) == 3:
        transformed_point = transformed_point.reshape(rows, cols, 3)
    
    return transformed_point

# Example usage
if __name__ == "__main__":
    # This is just to demonstrate how to use the functions
    # You would need to provide an actual image and config file
    import matplotlib.pyplot as plt
    
    # Example (replace with actual files)
    # image = cv2.imread('test_image.jpg')
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # belonging = rgb2belonging(image_rgb, 'color_configs.mat')
    # plt.imshow(belonging[:,:,0])  # Display first ellipsoid belonging
    # plt.show()
    
    print("rgb2belonging functions are ready to use")
