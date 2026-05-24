# inference.py
"""Run Model D on a single image and produce the Figure 1 composite.

Usage:
    python -m src.inference --image path/to/img.jpg --weights weights/model_D.pth --output_dir outputs/

The script loads the image, runs the depth head, builds a 3‑D bounding box (using placeholder intrinsics),
projects the box back to the image plane, renders a top‑down map, a 3‑D wireframe, and a point‑cloud silhouette.
The final four‑panel figure is saved as `figure1.png` inside the output directory.
"""

import argparse
import os
import cv2
import numpy as np
import torch
from .model_D import load_model
from .utils import (reconstruct_3d_points, project_to_image, draw_wireframe,
                    render_topdown_map, render_pointcloud_silhouette)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input RGB image")
    parser.add_argument("--weights", required=True, help="Path to Model D checkpoint")
    parser.add_argument("--output_dir", default="outputs", help="Where to store the figure")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    img = cv2.imread(args.image)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0).float() / 255.0

    model = load_model(args.weights)
    with torch.no_grad():
        depth = model(img_tensor).squeeze().cpu().numpy()

    # ---------- 3‑D reconstruction (placeholder values) ----------
    # Use a fixed camera intrinsics for demo purposes.
    cam_intr = {
        "fx": 800.0, "fy": 800.0,
        "cx": img.shape[1]/2, "cy": img.shape[0]/2,
        "height": 1.6, "pitch": 20.0, "roll": 0.0
    }
    points_3d = reconstruct_3d_points(depth, cam_intr)
    # Fit an oriented 3‑D box around the points (very simple AABB for demo)
    min_pt = points_3d.min(axis=0)
    max_pt = points_3d.max(axis=0)
    centre = (min_pt + max_pt) / 2
    dims = max_pt - min_pt
    yaw = 0.0  # placeholder orientation

    # ---------- Visualisation ----------
    # 1) Camera image with projected 3‑D box
    proj_corners = project_to_image(centre, dims, yaw, cam_intr)
    img_with_box = draw_wireframe(img_rgb.copy(), proj_corners, colour=(255,0,0))

    # 2) Top‑down map
    topdown = render_topdown_map(points_3d, cam_intr)

    # 3) 3‑D wireframe (static plot)
    wireframe_3d = draw_wireframe(None, proj_corners, colour=(0,255,0), three_d=True)

    # 4) Point‑cloud silhouette (2‑D projection)
    silhouette = render_pointcloud_silhouette(points_3d, cam_intr)

    # Assemble four panels into one composite image using OpenCV
    h, w = img.shape[:2]
    panel1 = cv2.cvtColor(img_with_box, cv2.COLOR_RGB2BGR)
    panel2 = cv2.cvtColor(topdown, cv2.COLOR_RGB2BGR)
    panel3 = cv2.cvtColor(wireframe_3d, cv2.COLOR_RGB2BGR)
    panel4 = cv2.cvtColor(silhouette, cv2.COLOR_RGB2BGR)
    top_row = np.hstack([panel1, panel2])
    bottom_row = np.hstack([panel3, panel4])
    composite = np.vstack([top_row, bottom_row])

    out_path = os.path.join(args.output_dir, "figure1.png")
    cv2.imwrite(out_path, composite)
    print(f"Figure 1 composite saved to {out_path}")

if __name__ == "__main__":
    main()
