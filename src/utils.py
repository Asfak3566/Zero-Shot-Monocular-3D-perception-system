# utils.py
"""Utility functions used by the Zero‑Shot Model D inference script.

All functions are lightweight and rely only on NumPy / OpenCV / Matplotlib.
They are deliberately simple – replace with your own high‑performance versions if needed.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def reconstruct_3d_points(depth: np.ndarray, intr: dict) -> np.ndarray:
    """Back‑project a depth map to a point cloud (Nx3).

    depth – (H, W) depth in meters.
    intr – dictionary with 'fx', 'fy', 'cx', 'cy'.
    """
    H, W = depth.shape
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    x = (i - intr['cx']) * depth / intr['fx']
    y = (j - intr['cy']) * depth / intr['fy']
    pts = np.stack([x, y, depth], axis=-1).reshape(-1, 3)
    # filter out far/zero points
    mask = (depth.reshape(-1) > 0) & (depth.reshape(-1) < 200)
    return pts[mask]

def project_to_image(centre, dims, yaw, intr):
    """Return 8 corners of an oriented 3‑D box projected to image coordinates.
    centre – (3,) world centre.
    dims – (3,) size (l, w, h).
    yaw – rotation around Z in radians.
    intr – same dict as above.
    """
    l, w, h = dims
    # define box corners in object frame
    corners = np.array([
        [ l/2,  w/2,  h/2], [ l/2, -w/2,  h/2],
        [-l/2, -w/2,  h/2], [-l/2,  w/2,  h/2],
        [ l/2,  w/2, -h/2], [ l/2, -w/2, -h/2],
        [-l/2, -w/2, -h/2], [-l/2,  w/2, -h/2]
    ])
    # rotation around Z
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    world_pts = (R @ corners.T).T + centre
    # project
    pts2d = []
    for p in world_pts:
        u = (p[0] * intr['fx']) / p[2] + intr['cx']
        v = (p[1] * intr['fy']) / p[2] + intr['cy']
        pts2d.append([u, v])
    return np.array(pts2d, dtype=int)

def draw_wireframe(img, corners, colour=(255,0,0), three_d=False):
    """Draw a 3‑D wireframe given projected corners.
    If *img* is None, creates a blank canvas of size 640×480.
    """
    if img is None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
    # box edges (same order as earlier scripts)
    edges = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)]
    for i, j in edges:
        cv2.line(img, tuple(corners[i]), tuple(corners[j]), colour, 2, lineType=cv2.LINE_AA)
    return img

def render_topdown_map(points, intr, size=(640,480)):
    """Render a simple top‑down occupancy map of the point cloud.
    Returns a RGB image.
    """
    canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    # orthographic projection (X-Z plane), scaling to image size
    xs = points[:,0]
    zs = points[:,2]
    # normalize to fit canvas
    min_x, max_x = xs.min(), xs.max()
    min_z, max_z = zs.min(), zs.max()
    scale_x = size[0] / (max_x - min_x + 1e-6)
    scale_z = size[1] / (max_z - min_z + 1e-6)
    u = ((xs - min_x) * scale_x).astype(int)
    v = ((zs - min_z) * scale_z).astype(int)
    cv2.circle(canvas, (u, v), 1, (0,255,0), -1)
    return canvas

def render_pointcloud_silhouette(points, intr, img_size=(640,480)):
    """Project the 3‑D points to the image plane and render as a silhouette.
    Returns a RGB image.
    """
    canvas = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
    for p in points:
        if p[2] <= 0: continue
        u = int((p[0] * intr['fx']) / p[2] + intr['cx'])
        v = int((p[1] * intr['fy']) / p[2] + intr['cy'])
        if 0 <= u < img_size[0] and 0 <= v < img_size[1]:
            canvas[v, u] = (255,255,255)
    return canvas
