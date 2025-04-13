import numpy as np
from equidistantpoints import EquidistantPoints

def generate_quaternion_data(num_samples=10000):
    """The idea is to have a grid of points sampled on the unit sphere
    and then convert them to quaternions. For validation we just use a random subset of the same grid.
    """

    points = EquidistantPoints(n_points=num_samples)
    cartesian_coordinates = np.array(points.cartesian)

    val_split = 0.1
    test_split = 0.2
    n_val = int(val_split * len(cartesian_coordinates))
    n_test = int(test_split * len(cartesian_coordinates))
    step = num_samples // n_val

    val_indices = np.arange(0, len(cartesian_coordinates), step)  # Take every 10th point for a uniform distribution
    train_indices = np.array([i for i in range(cartesian_coordinates.shape[0]) if i not in val_indices])
    
    # Generate quaternions
    quaternions = np.zeros((len(cartesian_coordinates), 4))
    points = cartesian_coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    # Ensure w is not negative or nan by clamping the value inside sqrt
    w = np.sqrt(np.maximum(0, 1 - x**2 - y**2 - z**2))
    # Normalize the quaternion to ensure unit length
    norm = np.sqrt(w**2 + x**2 + y**2 + z**2)
    quaternions = np.column_stack((w/norm, x/norm, y/norm, z/norm))

    val_set = quaternions[val_indices]
    train_test = quaternions[train_indices]

    # create a set of new data data are just randomly sampled on the sphere I don't care should be uniformely distributed
    # Generate random points in spherical coordinates
    theta = np.random.uniform(0, 2*np.pi, n_test)  # azimuthal angle
    phi = np.arccos(np.random.uniform(-1, 1, n_test))  # polar angle

    # Convert to Cartesian coordinates
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    # Stack into array of shape (1000, 3)
    test_set = np.column_stack((x, y, z))

    # convert to quaternions
    x = test_set[:, 0]
    y = test_set[:, 1]
    z = test_set[:, 2]
    # Ensure w is not negative or nan by clamping the value inside sqrt
    w = np.sqrt(np.maximum(0, 1 - x**2 - y**2 - z**2))
    norm = np.sqrt(w**2 + x**2 + y**2 + z**2)
    test_quaternions = np.column_stack((w / np.maximum(norm, 1e-10), x / np.maximum(norm, 1e-10), y / np.maximum(norm, 1e-10), z / np.maximum(norm, 1e-10)))

    return train_test, val_set, test_quaternions

def generate_translation_data(num_samples=10000):

    # Generate random translations
    translations = np.random.uniform(-1, 1, (num_samples, 3))

    val_split = 0.1
    test_split = 0.2
    n_val = int(val_split * len(translations))
    n_test = int(test_split * len(translations))
    step = num_samples // n_val

    val_indices = np.arange(0, len(translations), step)  # Take every 10th point for a uniform distribution
    train_indices = np.array([i for i in range(len(translations)) if i not in val_indices])

    val_set = translations[val_indices]
    train_test = translations[train_indices]

    # test The test set is just a random translation basically, just random numbers between -1 and 1
    test_set = np.random.uniform(-1, 1, (n_test, 3))

    return train_test, val_set, test_set

def generate_dataset(num_samples=10000):

    train_quaternions, val_quaternions, test_quaternions = generate_quaternion_data(num_samples=num_samples)
    train_translations, val_translations, test_translations = generate_translation_data(num_samples=num_samples)

    # we need to shuffle each dataset
    train_indices = np.random.permutation(len(train_quaternions))
    val_indices = np.random.permutation(len(val_quaternions))
    test_indices = np.random.permutation(len(test_quaternions))

    train_quaternions = train_quaternions[train_indices]
    val_quaternions = val_quaternions[val_indices]
    test_quaternions = test_quaternions[test_indices]

    # merge the two datasets
    train_set = np.concatenate((train_quaternions, train_translations), axis=1)
    val_set = np.concatenate((val_quaternions, val_translations), axis=1)
    test_set = np.concatenate((test_quaternions, test_translations), axis=1)

    return train_set, val_set, test_set