import numpy as np
from sklearn.cluster import DBSCAN
from scipy import stats
import matplotlib.pyplot as plt

def adust_local_t2nod(curr_pos, t2no, t2nd, t_max_occupancy):
    mask_idxs = localize_t2nod_tube(t2nod=t2no.T, eps=5, min_samples=10, max_t2nod=t_max_occupancy, curr_pos=(curr_pos[1], curr_pos[0]))
    
    t2nd[mask_idxs] = 0
    t2no[mask_idxs] = t_max_occupancy
    plt.imshow(t2no)

    return t2no.T, t2nd.T
    

def localize_t2nod_tube(t2nod, eps, min_samples, max_t2nod, curr_pos):
        """
        t2nod: t2no or t2nd, one of those
        returns indices of parts of t2nod to mask out
        """
        core_points, core_indices, labels, dbscan, filtered_t2nod = t2nod_DBSCAN(t2nod, eps, min_samples, max_t2nod)
        
        ego_idx = find_cluster(labels, core_points, core_indices, curr_pos, min_samples, radius=10, eps = eps)
        
        indices = filtered_t2nod[labels == ego_idx][:,:2]
        indices_tuple = tuple(np.array(indices).T)

        return indices_tuple

def t2nod_DBSCAN(t2nod, eps, min_samples, max_t2nod):
    """
    returns:
        - core_points:
        - core_indices: 
        - labels: output from fit function
        - dbscan: dbscan object
        - filtered_t2nod: data after processing 
    """

    # flatten data
    width, height = t2nod.shape[0], t2nod.shape[1]
    y_coords, x_coords = np.meshgrid(np.arange(width), np.arange(height))
    
    # Flatten arrays and stack them with intensity values
    x = x_coords.flatten()
    y = y_coords.flatten()

    intensity = t2nod.flatten()

    # Combine x, y, intensity into a single array
    pixel_data = np.column_stack((y,x, intensity))

    filtered_t2nod = pixel_data[pixel_data[:,2] != max_t2nod] # remove max_t2nod_plane
    
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(filtered_t2nod)

    core_indices = dbscan.core_sample_indices_
    core_points = filtered_t2nod[core_indices]

    return core_points, core_indices, labels, dbscan, filtered_t2nod


def generate_points_within_circle(center_x, center_y, radius, num_points):
    points = []
    while len(points) < num_points:
        # Generate a random point within the circle
        x = np.random.randint(center_x - radius, center_x + radius + 1)
        y_range = np.sqrt(radius**2 - (x - center_x)**2)
        y = np.random.randint(max(center_y - y_range, center_y - radius),
                              min(center_y + y_range + 1, center_y + radius))

        new_point = (x, y)
        
        # Check if the new point is too close to existing points
        if all(np.linalg.norm(np.array(new_point) - np.array(existing_point)) >= 1 for existing_point in points):
            points.append(new_point)

    return np.array(points)

def find_cluster(labels, core_points, core_indices, current_pos, sample_size = 20, radius = 10, eps = 5):
    """ Finds which index the given point belongs to"""

    samples = generate_points_within_circle(current_pos[0], current_pos[1], radius, sample_size)
    distances = np.linalg.norm(samples[:, np.newaxis, :] - core_points[:,:2], axis=2)
    min_distances = np.min(distances, axis=1)
    nearest_core_point_indices = np.argmin(distances, axis=1)

    cluster_indices = min_distances <= eps
    cluster_labels = labels[core_indices[nearest_core_point_indices[cluster_indices]]]
    ego_idx = stats.mode(cluster_labels).mode

    return ego_idx