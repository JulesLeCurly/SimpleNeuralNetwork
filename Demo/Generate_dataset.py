import numpy as np

def generate_circle_dataset(num_samples=200, noise=0.1):
    """
    Generate a circle dataset for binary classification.
    
    Parameters:
    - num_samples: total number of samples
    - noise: noise level (0 to 1)
    
    Returns:
    - X: numpy array of shape (num_samples, 2) with coordinates
    - y: numpy array of shape (num_samples,) with labels (1 or -1)
    """
    points = []
    radius = 5
    
    def get_circle_label(x, y):
        dist = np.sqrt(x**2 + y**2)
        return 1 if dist < (radius * 0.5) else -1
    
    # Generate positive points inside the circle
    for i in range(num_samples // 2):
        r = np.random.uniform(0, radius * 0.5)
        angle = np.random.uniform(0, 2 * np.pi)
        x = r * np.sin(angle)
        y = r * np.cos(angle)
        noise_x = np.random.uniform(-radius, radius) * noise
        noise_y = np.random.uniform(-radius, radius) * noise
        label = get_circle_label(x + noise_x, y + noise_y)
        points.append([x, y, label])
    
    # Generate negative points outside the circle
    for i in range(num_samples // 2):
        r = np.random.uniform(radius * 0.7, radius)
        angle = np.random.uniform(0, 2 * np.pi)
        x = r * np.sin(angle)
        y = r * np.cos(angle)
        noise_x = np.random.uniform(-radius, radius) * noise
        noise_y = np.random.uniform(-radius, radius) * noise
        label = get_circle_label(x + noise_x, y + noise_y)
        points.append([x, y, label])
    
    points = np.array(points)
    X = points[:, :2]
    y = points[:, 2]

    X = X.astype(np.float16)
    y = y.astype(np.int8)

    X = (X -X.min()) / (X.max() - X.min())
    
    y = np.where(y == -1, 0, y).reshape(-1, 1)

    return X, y
