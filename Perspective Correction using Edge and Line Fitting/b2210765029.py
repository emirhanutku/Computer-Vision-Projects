#!/usr/bin/env python
# coding: utf-8

# # BBM418 - Computer Vision Lab Assignment 1
# # Student Name: Emirhan Utku
# # Student ID: 2210765029
# # Date: March 30, 2025

# In[89]:


import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import structural_similarity as ssim
import random
import matplotlib.pyplot as plt


# # -----------------------------------------------
# # 0. Helper Function for RANSAC
# # -----------------------------------------------

# In[90]:


def average_distance_in_image(line1, line2, width, height, n_samples=10):
  
    (A1, B1, C1) = line1
    (A2, B2, C2) = line2
    
    distances = []
    
    step_x = max(1, width // n_samples)
    for x in range(0, width, step_x):
        # If B1 is not too small, we can solve y = -(A1*x + C1)/B1
        if abs(B1) > 1e-6:
            y = -(A1*x + C1)/B1
            if 0 <= y < height:
                # distance from (x,y) to line2 is |A2*x + B2*y + C2|
                dist_pt_line2 = abs(A2*x + B2*y + C2)
                distances.append(dist_pt_line2)
    
    step_y = max(1, height // n_samples)
    for y in range(0, height, step_y):
        # If A1 is not too small, we can solve x = -(B1*y + C1)/A1
        if abs(A1) > 1e-6:
            x = -(B1*y + C1)/A1
            if 0 <= x < width:
                dist_pt_line2 = abs(A2*x + B2*y + C2)
                distances.append(dist_pt_line2)
    
    if len(distances) == 0:
        # If we couldn't sample anything, return a large number so it won't be considered "similar"
        return 1e6
    
    return sum(distances) / len(distances)


# In[91]:


def double_check_refined_lines(refined_lines, image_shape, geom_thresh=10, n_samples=10):

    h, w = image_shape[:2]
    final_lines = []
    for line in refined_lines:
        A, B, C, inlier_count = line
        merged = False
        for idx, existing_line in enumerate(final_lines):
            Ae, Be, Ce, inlier_e = existing_line
            avg_dist = average_distance_in_image((A, B, C), (Ae, Be, Ce), width=w, height=h, n_samples=n_samples)
            if avg_dist < geom_thresh:
                # If lines are geometrically similar, keep the one with the higher inlier count.
                if inlier_count > inlier_e:
                    final_lines[idx] = line
                merged = True
                break
        if not merged:
            final_lines.append(line)
    return final_lines


# # -----------------------------------------------
# # 1. Preprocessing and Edge Detection
# # -----------------------------------------------

# In[94]:


def load_and_resize(image_path, scale_factor=0.3):
    image = cv2.imread(image_path)
    
    if image is None:
         raise ValueError("Image not loaded: " + image_path)
            
    resized = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
    return resized

def edge_detection(image):
    blur = 5
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (31, 31), blur)
    
    median = np.median(blurred_image)
    lower, upper = int(max(0, 0.66 * median))+20, int(min(255, 1.33 * median))+20
    edges = cv2.Canny(blurred_image, lower, upper)

    
    while(True):
        if np.sum(edges > 0) < 4_000:
            blur -= 0.05
            blurred_image = cv2.GaussianBlur(gray_image, (31, 31), blur)
            edges = cv2.Canny(blurred_image, lower, upper)
        
        else:
            return edges
    


# # -----------------------------------------------
# # 2. Hough Transform Implementation
# # -----------------------------------------------

# In[95]:


def hough_transform(edges, rho_resolution=1, theta_resolution=np.pi/180):
    height, width = edges.shape
    diag_len = int(np.sqrt(height**2 + width**2))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)
    thetas = np.arange(0, np.pi, theta_resolution)
    
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(edges)  
    

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(len(thetas)):
            theta = thetas[t_idx]
            rho = int(round(x * np.cos(theta) + y * np.sin(theta))) + diag_len
            accumulator[rho, t_idx] += 1
            
    return accumulator, rhos, thetas


# In[96]:


def detect_lines_from_hough(accumulator, rhos, thetas, fraction=0.1, min_lines=50, max_lines=75):

    peaks = []
    for i in range(accumulator.shape[0]):
        for j in range(accumulator.shape[1]):
            if accumulator[i, j] > 0:
                peaks.append((accumulator[i, j], rhos[i], thetas[j]))
    
    # Sort the candidate lines in descending order by their vote counts.
    peaks.sort(key=lambda x: x[0], reverse=True)
    
    # Extract the (rho, theta) for each candidate line.
    candidate_lines = [(rho, theta) for (votes, rho, theta) in peaks]
    n = len(candidate_lines)
    
    if n == 0:
        return []
    

    k = int(fraction * n)

    if k < min_lines:
        k = min(min_lines, n)

        elif k > max_lines:
        k = max_lines
        
    return candidate_lines[:k]


# # -----------------------------------------------
# # 3. RANSAC for Robust Line Fitting
# # -----------------------------------------------

# In[97]:


def normalize_line(A, B, C):

    norm = np.sqrt(A**2 + B**2)
    if norm < 1e-9:
        return (A, B, C)
    A /= norm
    B /= norm
    C /= norm

    if (A < 0) or (abs(A) < 1e-9 and B < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def line_angle_dist(A, B, C):

    angle = np.arctan2(B, A)
    dist = -C
    return angle, dist

def is_similar_line(line1, line2, angle_thresh_deg=5, dist_thresh=10):

    (A1, B1, C1) = line1
    (A2, B2, C2) = line2
    angle1, d1 = line_angle_dist(A1, B1, C1)
    angle2, d2 = line_angle_dist(A2, B2, C2)
    

    angle_diff = abs(angle1 - angle2)
    angle_diff = min(angle_diff, 2*np.pi - angle_diff)
    angle_diff_deg = np.degrees(angle_diff)
    
    dist_diff = abs(d1 - d2)


    return (angle_diff_deg < angle_thresh_deg) and (dist_diff < dist_thresh)


# In[98]:


def get_points_near_line(edges, rho, theta, distance_thresh=1.0):

    y_idxs, x_idxs = np.nonzero(edges)
    points = np.column_stack((x_idxs, y_idxs))  # shape: (N, 2)
    
    
    distances = np.abs(points[:, 0]*np.cos(theta) + points[:, 1]*np.sin(theta) - rho)
    inlier_mask = distances < distance_thresh
    return points[inlier_mask]

def ransac_line_fitting(points, num_iterations=1000, distance_threshold=2.0):

    best_inlier_count = 0
    best_line = None
    n_points = points.shape[0]
    
    if n_points < 2:
        return None
    
    for _ in range(num_iterations):
        idx = np.random.choice(n_points, 2, replace=False)
        p1, p2 = points[idx]
        

        A = p2[1] - p1[1]
        B = p1[0] - p2[0]
        C = p2[0]*p1[1] - p1[0]*p2[1]
        

        (A, B, C) = normalize_line(A, B, C)
        
        # Distances to line
        distances = np.abs(A*points[:, 0] + B*points[:, 1] + C)
        inlier_mask = distances < distance_threshold
        inlier_count = np.sum(inlier_mask)
        
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_line = (A, B, C, inlier_count)
    
    return best_line

def draw_line(img, line, color=(0, 255, 0), thickness=2):

    A, B, C = line
    h, w = img.shape[:2]
    if np.abs(B) > 1e-6:

        pt1 = (0, int(-C/B))
        pt2 = (w, int((-C - A*w)/B))
    else:
        x = int(-C/A)
        pt1 = (x, 0)
        pt2 = (x, h)
    cv2.line(img, pt1, pt2, color, thickness)


# In[99]:


def pick_rectangle_lines(refined_lines):

    
    horizontal_lines = []
    vertical_lines = []
    for (A, B, C, inlier) in refined_lines:
        
        angle_rad = np.arctan(-(A/B))             
        angle_deg = np.degrees(angle_rad) 
        angle = angle_deg % 180

        if angle < 25 or angle > 151:
            if abs(B) < 1e-6:
                continue
            y_int = -C / B  
            horizontal_lines.append(((A, B, C, inlier), y_int))

        elif 75 < angle <105:
            if abs(A) < 1e-6:
                continue
            x_int = -C / A  
            vertical_lines.append(((A, B, C, inlier), x_int))
    

    if len(horizontal_lines) >= 2:
    
        horizontal_lines.sort(key=lambda x: x[1])
        top_line = horizontal_lines[0][0]
        bottom_line = horizontal_lines[-1][0]
       
    else:
        top_line = None
        bottom_line = None

    if len(vertical_lines) >= 2:

        vertical_lines.sort(key=lambda x: x[1])
        left_line = vertical_lines[0][0]
        right_line = vertical_lines[-1][0]
        
     
    else:
        left_line = None
        right_line = None

    rectangle_lines = []
    if top_line is not None:
        rectangle_lines.append(top_line)
    if bottom_line is not None:
        rectangle_lines.append(bottom_line)
    if left_line is not None:
        rectangle_lines.append(left_line)
    if right_line is not None:
        rectangle_lines.append(right_line)
        
        
    if len(rectangle_lines)< 4:
        print("No lines were found to form a quadritale.")
        return refined_lines
    
    return rectangle_lines


# # ---------------------------
# # 4. Quadrilateral Detection from Refined Lines
# # ---------------------------

# In[100]:


def compute_intersection(line1, line2):

    A1, B1, C1 = line1
    A2, B2, C2 = line2
    det = A1*B2 - A2*B1
    if abs(det) < 1e-6:
        return None  
    x = (B2 * (-C1) - B1 * (-C2)) / det
    y = (A1 * (-C2) - A2 * (-C1)) / det
    return (x, y)

def find_document_quadrilateral(refined_lines, image_shape):

    intersections = []
    h, w = image_shape[:2]
    
    for i in range(len(refined_lines)):
        for j in range(i+1, len(refined_lines)):
            line1 = refined_lines[i][:3]  # (A, B, C)
            line2 = refined_lines[j][:3]
            
            angle1 = np.arctan2(line1[1], line1[0])
            angle2 = np.arctan2(line2[1], line2[0])
            angle_diff = abs(angle1 - angle2)
            angle_diff = min(angle_diff, np.pi - angle_diff)
            if angle_diff < np.deg2rad(10): 
                continue
            
            pt = compute_intersection(line1, line2)
            if pt is None:
                continue
            x, y = pt
            if 0 <= x <= w and 0 <= y <= h:
                intersections.append(pt)
                
    intersections = np.array(intersections, dtype=np.float32)
    if intersections.shape[0] == 0:
        return None  
    
    # Compute the convex hull of the intersections.
    hull = cv2.convexHull(intersections)
    
    # Approximate the convex hull polygon.
    epsilon = 0.05 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    
    if len(approx) == 4:

        quad = [tuple(pt[0]) for pt in approx]
        return quad
    else:
        # As a fallback, return the hull points.
        return [tuple(pt[0]) for pt in hull]


# 
# # ---------------------------
# # 4. Custom Perspective Transform (Homography + Vectorized Bilinear Interpolation)
# # ---------------------------

# In[101]:


def order_points(pts):

    rect = np.zeros((4, 2), dtype="float64")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect


# In[102]:


def compute_homography(src_pts, dst_pts):

    assert len(src_pts) == 4 and len(dst_pts) == 4, "Need exactly 4 source and 4 destination points."
    
    # Build the 8x9 matrix A (2 rows per correspondence)
    A = []
    for i in range(4):
        x, y = src_pts[i]
        u, v = dst_pts[i]
        A.append([-x, -y, -1,   0,   0,   0,   x*u, y*u, u])
        A.append([ 0,   0,   0,  -x, -y,  -1,   x*v, y*v, v])
    A = np.array(A, dtype=np.float64)  # shape (8,9)
    
    # Solve A * h = 0 using SVD. The solution is the last column of V (or last row of V^T).
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1]  # eigenvector with smallest eigenvalue
    
    # Reshape to 3x3
    H = h.reshape(3, 3)
    
    # Normalize so that H[2,2] = 1 if it's not 0
    if abs(H[2,2]) > 1e-8:
        H /= H[2,2]
    
    return H


# In[103]:


def warp_image_bilinear(src_image, H, out_width, out_height):

    
    H_inv = np.linalg.inv(H)
    
    if len(src_image.shape) == 3:
        h_src, w_src, c_src = src_image.shape
        warped = np.zeros((out_height, out_width, c_src), dtype=src_image.dtype)
    else:
        h_src, w_src = src_image.shape
        c_src = 1
        warped = np.zeros((out_height, out_width), dtype=src_image.dtype)
    
    
    X, Y = np.meshgrid(np.arange(out_width), np.arange(out_height))
    
    
    X_flat = X.flatten()  
    Y_flat = Y.flatten()  
    ones = np.ones_like(X_flat) 
    

    dst_pts = np.stack([X_flat, Y_flat, ones], axis=0)  
    
    src_pts = H_inv @ dst_pts  
    
    
    x_prime = src_pts[0] / src_pts[2]  
    y_prime = src_pts[1] / src_pts[2]
    
    # 3) Bilinear interpolation
    x0 = np.floor(x_prime).astype(int)
    y0 = np.floor(y_prime).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    
    alpha = x_prime - x0
    beta = y_prime - y0
    
    # Valid range check
    valid_mask = (x0 >= 0) & (x1 < w_src) & (y0 >= 0) & (y1 < h_src)
    
    # For convenience, define a function to safely gather pixel values
    def gather_values(img, xx, yy):
        xx = np.clip(xx, 0, img.shape[1] - 1)
        yy = np.clip(yy, 0, img.shape[0] - 1)
        return img[yy, xx]

    if c_src > 1:
        # Color image, shape: H_src x W_src x C
        I_y0x0 = gather_values(src_image, x0, y0)
        I_y0x1 = gather_values(src_image, x1, y0)
        I_y1x0 = gather_values(src_image, x0, y1)
        I_y1x1 = gather_values(src_image, x1, y1)
        
        # Expand alpha, beta to (N,1) so broadcasting works for color channels
        alpha_2d = alpha[:, np.newaxis]
        beta_2d = beta[:, np.newaxis]
        
        out_val = (1 - alpha_2d)*(1 - beta_2d)*I_y0x0                   + alpha_2d*(1 - beta_2d)*I_y0x1                   + (1 - alpha_2d)*beta_2d*I_y1x0                   + alpha_2d*beta_2d*I_y1x1
        
        # Zero out invalid
        out_val[~valid_mask] = 0
        
        # Reshape back to (out_height, out_width, C)
        warped = out_val.reshape((out_height, out_width, c_src))
    else:
        # Grayscale image, shape: H_src x W_src
        I_y0x0 = gather_values(src_image, x0, y0)
        I_y0x1 = gather_values(src_image, x1, y0)
        I_y1x0 = gather_values(src_image, x0, y1)
        I_y1x1 = gather_values(src_image, x1, y1)
        
        out_val = (1 - alpha)*(1 - beta)*I_y0x0                   + alpha*(1 - beta)*I_y0x1                   + (1 - alpha)*beta*I_y1x0                   + alpha*beta*I_y1x1
        
        out_val[~valid_mask] = 0
        warped = out_val.reshape((out_height, out_width))
    
    return warped


# In[104]:


def four_point_transform_manual_bilinear(src_image, src_pts):

    # 1) Order corners
    rect = order_points(src_pts)  # [top-left, top-right, bottom-right, bottom-left]
    (tl, tr, br, bl) = rect

    # 2) Compute the new image width and height
    widthA = np.sqrt((br[0] - bl[0])**2 + (br[1] - bl[1])**2)
    widthB = np.sqrt((tr[0] - tl[0])**2 + (tr[1] - tl[1])**2)
    maxWidth = int(max(widthA, widthB))

    heightA = np.sqrt((tr[0] - br[0])**2 + (tr[1] - br[1])**2)
    heightB = np.sqrt((tl[0] - bl[0])**2 + (tl[1] - bl[1])**2)
    maxHeight = int(max(heightA, heightB))

    # 3) Destination corners for a "tight" rectangle
    dst_pts = np.array([
        [0,           0],
        [maxWidth-1,  0],
        [maxWidth-1,  maxHeight-1],
        [0,           maxHeight-1]
    ], dtype=np.float64)

    # 4) Compute homography from rect -> dst_pts
    H = compute_homography(rect, dst_pts)

    # 5) Warp using bilinear interpolation
    warped = warp_image_bilinear(src_image, H, maxWidth, maxHeight)
    return warped


# # -----------------------------------------------
# # 5. SSIM Evaluation
# # -----------------------------------------------

# In[105]:


def compute_ssim(image1, image2):
    # Convert to 8-bit if needed
    if image1.dtype != np.uint8:
        image1 = cv2.convertScaleAbs(image1)
    if image2.dtype != np.uint8:
        image2 = cv2.convertScaleAbs(image2)
    
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score


# # -----------------------------------------------
# # 6. Dataset Processing
# # -----------------------------------------------

# In[107]:


def process_and_save_sample(image_path, gt_path,ransac_dist_threshold, scale_factor=0.3):
    
    # Load full-resolution images
    orig_image = cv2.imread(image_path)
    if orig_image is None:
        raise ValueError("Original image not loaded: " + image_path)
    orig_gt_image = cv2.imread(gt_path)
    if orig_gt_image is None:
        raise ValueError("Ground truth image not loaded: " + gt_path)
   
    # Create a rescaled copy for processing
   
    image = load_and_resize(image_path, scale_factor)
    gt_image = load_and_resize(gt_path, scale_factor)
    
    # Edge Detection on rescaled image
    edges = edge_detection(image)
    cv2.imwrite("sample_edges.jpg", edges)
    print("Edges saved")
    
    # Hough Transform to detect candidate lines on rescaled image
    accumulator, rhos, thetas = hough_transform(edges)
    candidate_lines = detect_lines_from_hough(accumulator, rhos, thetas)
    print("Number of candidate lines from Hough:", len(candidate_lines))
   
    
    refined_lines = []  # Each element: (A, B, C, inlier_count)
    for (rho, theta) in candidate_lines:
        line_points = get_points_near_line(edges, rho, theta, distance_thresh=2.0)
        if len(line_points) < 2:
            continue
        best_line = ransac_line_fitting(line_points, num_iterations=2500, distance_threshold=1.5)

        if best_line is None:
            continue
        (A, B, C, inlier_count) = best_line

        new_line_normalized = normalize_line(A, B, C)
        skip = False
        for line_existing in refined_lines:
            (Ae, Be, Ce, inlier_e) = line_existing
            existing_normalized = (Ae, Be, Ce)
            if is_similar_line(new_line_normalized, existing_normalized, angle_thresh_deg=45, dist_thresh=ransac_dist_threshold):
                skip = True
                if inlier_count > inlier_e:
                    refined_lines.remove(line_existing)
                    refined_lines.append((A, B, C, inlier_count))
                break
        if not skip:
            refined_lines.append((A, B, C, inlier_count))
    
    refined_lines = double_check_refined_lines(refined_lines, image_shape=edges.shape,
                                           geom_thresh=85, n_samples=10)
    
    if(len(refined_lines)>4):
        print("There are ",len(refined_lines)," lines. Checking again because there are more than 4 lines...")
        refined_lines = pick_rectangle_lines(refined_lines)
        
    print(f"Refined lines count (after filtering): {len(refined_lines)}")
  
    
    # Draw refined lines on rescaled image
    image_with_lines = image.copy()
    for (A, B, C, inlier_count) in refined_lines:
        draw_line(image_with_lines, (A, B, C), color=(0, 0, 255), thickness=2)
    cv2.imwrite("sample_refined_lines.jpg", image_with_lines)
    print("Refined lines image saved as sample_refined_lines.jpg")
    
    # Also draw candidate Hough lines (blue) for comparison
    image_hough = image.copy()
    for rho, theta in candidate_lines:
        a = np.cos(theta)
        b = np.sin(theta)
        c = -rho
        draw_line(image_hough, (a, b, c), color=(255, 0, 0), thickness=2)
    cv2.imwrite("sample_hough_lines.jpg", image_hough)
    

    quad = find_document_quadrilateral(refined_lines, image.shape)
    if quad is not None and len(quad) == 4:
        print("Quadrilateral corners found:", quad)
        for pt in quad:
            cv2.circle(image_with_lines, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
    else:
        print("Quadrilateral not detected. Returned points:", quad)
    cv2.imwrite("sample_corners.jpg", image_with_lines)
    print("Refined lines image with corners saved as sample_corners.jpg")
    

    if quad is not None and len(quad) == 4:
        scale = 1.0 / scale_factor
        quad_original = [(pt[0]*scale, pt[1]*scale) for pt in quad]
        
        
        quad_np = np.array(quad_original, dtype=np.float64)
        warped = four_point_transform_manual_bilinear(orig_image, quad_np)
        if warped is not None and warped.size!=0:
            
            cv2.imwrite("warped_result.jpg", warped)
            print("Saved warped (rectified) document as warped_result.jpg")
        
        else:
            warped= None
    else:
        warped = None
        print("Could not perform perspective transform due to missing quadrilateral.")
    
    # Compute SSIM between warped full-resolution image and full-resolution ground truth
    if warped is not None:
        if warped.shape != orig_gt_image.shape:
            warped_for_ssim = cv2.resize(warped, (orig_gt_image.shape[1], orig_gt_image.shape[0]))
        else:
            warped_for_ssim = warped
        ssim_value = compute_ssim(orig_gt_image, warped_for_ssim)
        print("SSIM between warped image and ground truth:", ssim_value)
    else:
        disorted_image = cv2.imread(image_path)
        
        if disorted_image.shape != orig_gt_image.shape:
            disorted_image = cv2.resize(disorted_image, (orig_gt_image.shape[1], orig_gt_image.shape[0]))

        ssim_value = compute_ssim(orig_gt_image, disorted_image)
    
    return {
        "edges": edges,
        "hough_lines_image": image_hough,
        "ransac_lines_image": image_with_lines,
        "corrected": warped,
        "ssim": ssim_value
    }


# # -----------------------------------------------
# # 7. Calculate Results
# # -----------------------------------------------

# In[108]:


def calculate_results_and_ssim_for_folder(distorted_folder, gt_folder,ransac_dist_threshold,folder_name, scale_factor=0.25, num_images=50):

    distorted_files = [os.path.join(distorted_folder, f)
                       for f in os.listdir(distorted_folder)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Shuffle and select a subset of images
    random.shuffle(distorted_files)
    selected_files = distorted_files[:num_images]
    
    full_results = {}
    ssim_results = {}
    
    for i, file_path in enumerate(selected_files, start=1):
        # Construct the corresponding ground-truth image path (assuming same filename)
        gt_path = os.path.join(gt_folder, os.path.basename(file_path))
        print(f"------------------------- IMAGE {i} START ------------------------------------")
        print(f"Processing image: {file_path}")
        
        # Call process_and_save_sample and get the full result dictionary
        result = process_and_save_sample(file_path, gt_path, ransac_dist_threshold, scale_factor=scale_factor)
        full_results[file_path] = result
        
        # Store the SSIM value from the result
        ssim_results[file_path] = result["ssim"]
        print(f"Image: {file_path}, SSIM: {result['ssim']}")
        print(f"------------------------- IMAGE {i} END ------------------------------------")
    
    # Compute the average SSIM value
    average_ssim = sum(ssim_results.values()) / len(ssim_results) if ssim_results else 0
    
    return full_results, average_ssim


# # -----------------------------------------------
# # 8. Function that can be used to control images one by one
# # -----------------------------------------------

# In[145]:


sample_distorted_image = 'WarpDoc/distorted/rotate/0043.jpg'
sample_ground_truth_image = 'WarpDoc/digital/rotate/0043.jpg'


# Process the sample image and save outputs as JPG files
outputs = process_and_save_sample(sample_distorted_image, sample_ground_truth_image,125, scale_factor=0.3)


# # -----------------------------------------------
# # 9. Results for Each Folder
# # -----------------------------------------------

# # COMPUTE AVERAGE SSIM FOR CURVED FOLDER

# ### Ransac threshold : 125

# In[21]:


full_results_curved, average_ssim_curved = calculate_results_and_ssim_for_folder(
         'WarpDoc/distorted/curved/', 
         'WarpDoc/digital/curved/',
         125,
         "Curved folder",
         scale_factor=0.3, 
         num_images=50
     )
print("----------------------- SSIM calculation with ", 125," ransac distance threshold is finished -----------------------")


# In[22]:


print("------------------------------------------------------------------------------")
print(f"\nAverage SSIM of the Curved folder with a RANSAC distance threshold of 125 : {average_ssim_curved:.4f}")


# ### Ransac threshold : 150

# In[23]:


full_results_curved, average_ssim_curved = calculate_results_and_ssim_for_folder(
         'WarpDoc/distorted/curved/', 
         'WarpDoc/digital/curved/',
         150,
         "Curved folder",
         scale_factor=0.3, 
         num_images=50
     )
print("----------------------- SSIM calculation with ", 150," ransac distance threshold is finished -----------------------")


# In[24]:


print("------------------------------------------------------------------------------")
print(f"\nAverage SSIM of the Curved folder with a RANSAC distance threshold of 150 : {average_ssim_curved:.4f}")


# ### Ransac threshold : 200

# In[25]:


full_results_curved, average_ssim_curved = calculate_results_and_ssim_for_folder(
         'WarpDoc/distorted/curved/', 
         'WarpDoc/digital/curved/',
         200,
         "Curved folder",
         scale_factor=0.3, 
         num_images=50
     )
print("----------------------- SSIM calculation with ", 200," ransac distance threshold is finished -----------------------")


# In[26]:


print("------------------------------------------------------------------------------")
print(f"\nAverage SSIM of the Curved folder with a RANSAC distance threshold of 200 : {average_ssim_curved:.4f}")


# In[ ]:





# In[ ]:





# ----

# # COMPUTE AVERAGE SSIM FOR FOLD FOLDER

# ### Ransac threshold : 125

# In[27]:


full_results_fold, average_ssim_fold = calculate_results_and_ssim_for_folder(
         'WarpDoc/distorted/fold/', 
         'WarpDoc/digital/fold/',
         125,
         "Fold folder",
         scale_factor=0.3, 
         num_images=50
     )
print("----------------------- SSIM calculation with ", 125," ransac distance threshold is finished -----------------------")


# In[28]:


print("------------------------------------------------------------------------------")
print(f"\nAverage SSIM of the Fold folder with a RANSAC distance threshold of 125 : {average_ssim_fold:.4f}")


# ### Ransac threshold : 150

# In[29]:


full_results_fold, average_ssim_fold = calculate_results_and_ssim_for_folder(
         'WarpDoc/distorted/fold/', 
         'WarpDoc/digital/fold/',
         150,
         "Fold folder",
         scale_factor=0.3, 
         num_images=50
     )
print("----------------------- SSIM calculation with ", 150," ransac distance threshold is finished -----------------------")


# In[30]:


print("------------------------------------------------------------------------------")
print(f"\nAverage SSIM of the Fold folder with a RANSAC distance threshold of 150 : {average_ssim_fold:.4f}")


# ### Ransac threshold : 200

# In[31]:


full_results_fold, average_ssim_fold = calculate_results_and_ssim_for_folder(
         'WarpDoc/distorted/fold/', 
         'WarpDoc/digital/fold/',
         200,
         "Fold folder",
         scale_factor=0.3, 
         num_images=50
     )
print("----------------------- SSIM calculation with ", 200," ransac distance threshold is finished -----------------------")


# In[32]:


print("------------------------------------------------------------------------------")
print(f"\nAverage SSIM of the Fold folder with a RANSAC distance threshold of 200 : {average_ssim_fold:.4f}")


# In[ ]:





# In[ ]:





# -----

# # COMPUTE AVERAGE SSIM FOR INCOMPLETE FOLDER

# ### Ransac threshold : 125

# In[33]:


full_results_incomplete, average_ssim_incomplete = calculate_results_and_ssim_for_folder(
         'WarpDoc/distorted/incomplete/', 
         'WarpDoc/digital/incomplete/',
         125,
         "Incomplete folder",
         scale_factor=0.3, 
         num_images=50
     )
print("----------------------- SSIM calculation with ", 125," ransac distance threshold is finished -----------------------")


# In[34]:


print("------------------------------------------------------------------------------")
print(f"\nAverage SSIM of the Incomplete folder with a RANSAC distance threshold of 125 : {average_ssim_incomplete:.4f}")


# ### Ransac threshold : 150

# In[35]:


full_results_incomplete, average_ssim_incomplete = calculate_results_and_ssim_for_folder(
         'WarpDoc/distorted/incomplete/', 
         'WarpDoc/digital/incomplete/',
         150,
         "Incomplete folder",
         scale_factor=0.3, 
         num_images=50
     )
print("----------------------- SSIM calculation with ", 150," ransac distance threshold is finished -----------------------")


# In[36]:


print("------------------------------------------------------------------------------")
print(f"\nAverage SSIM of the Incomplete folder with a RANSAC distance threshold of 150 : {average_ssim_incomplete:.4f}")


# ## Ransac threshold : 200

# In[37]:


full_results_incomplete, average_ssim_incomplete = calculate_results_and_ssim_for_folder(
         'WarpDoc/distorted/incomplete/', 
         'WarpDoc/digital/incomplete/',
         200,
         "Incomplete folder",
         scale_factor=0.3, 
         num_images=50
     )
print("----------------------- SSIM calculation with ", 200," ransac distance threshold is finished -----------------------")


# In[38]:


print("------------------------------------------------------------------------------")
print(f"\nAverage SSIM of the Incomplete folder with a RANSAC distance threshold of 200 : {average_ssim_incomplete:.4f}")


# -------

# In[ ]:





# In[ ]:





# # COMPUTE AVERAGE SSIM FOR PERSPECTİVE FOLDER

# ### Ransac threshold : 125

# In[39]:


full_results_perspective, average_ssim_perspective = calculate_results_and_ssim_for_folder(
         'WarpDoc/distorted/perspective/', 
         'WarpDoc/digital/perspective/',
         125,
         "Perspective folder",
         scale_factor=0.3, 
         num_images=50
     )
print("----------------------- SSIM calculation with ", 125," ransac distance threshold is finished -----------------------")


# In[40]:


print("------------------------------------------------------------------------------")
print(f"\nAverage SSIM of the Perspective folder with a RANSAC distance threshold of 125 : {average_ssim_perspective:.4f}")



# ### Ransac threshold : 150

# In[41]:


full_results_perspective, average_ssim_perspective = calculate_results_and_ssim_for_folder(
         'WarpDoc/distorted/perspective/', 
         'WarpDoc/digital/perspective/',
         150,
         "Perspective folder",
         scale_factor=0.3, 
         num_images=50
     )
print("----------------------- SSIM calculation with ", 150," ransac distance threshold is finished -----------------------")


# In[42]:


print("------------------------------------------------------------------------------")
print(f"\nAverage SSIM of the Perspective folder with a RANSAC distance threshold of 150 : {average_ssim_perspective:.4f}")


# ### Ransac threshold : 200

# In[43]:


full_results_perspective, average_ssim_perspective = calculate_results_and_ssim_for_folder(
         'WarpDoc/distorted/perspective/', 
         'WarpDoc/digital/perspective/',
         200,
         "Perspective folder",
         scale_factor=0.3, 
         num_images=50
     )
print("----------------------- SSIM calculation with ", 200," ransac distance threshold is finished -----------------------")


# In[44]:


print("------------------------------------------------------------------------------")
print(f"\nAverage SSIM of the Perspective folder with a RANSAC distance threshold of 200 : {average_ssim_perspective:.4f}")


# In[ ]:





# In[ ]:





# # COMPUTE AVERAGE SSIM FOR RANDOM FOLDER

# ### Ransac threshold : 125

# In[47]:


full_results_random, average_ssim_random = calculate_results_and_ssim_for_folder(
         'WarpDoc/distorted/random/', 
         'WarpDoc/digital/random/',
         125,
         "Random folder",
         scale_factor=0.3, 
         num_images=50
     )
print("----------------------- SSIM calculation with ", 125," ransac distance threshold is finished -----------------------")


# In[48]:


print("------------------------------------------------------------------------------")
print(f"\nAverage SSIM of the Random folder with a RANSAC distance threshold of 125 : {average_ssim_random:.4f}")



# ### Ransac threshold : 150

# In[49]:


full_results_random, average_ssim_random = calculate_results_and_ssim_for_folder(
         'WarpDoc/distorted/random/', 
         'WarpDoc/digital/random/',
         150,
         "Random folder",
         scale_factor=0.3, 
         num_images=50
     )
print("----------------------- SSIM calculation with ", 150," ransac distance threshold is finished -----------------------")


# In[50]:


print("------------------------------------------------------------------------------")
print(f"\nAverage SSIM of the Random folder with a RANSAC distance threshold of 150 : {average_ssim_random:.4f}")


# ### Ransac threshold : 200

# In[51]:


full_results_random, average_ssim_random = calculate_results_and_ssim_for_folder(
         'WarpDoc/distorted/random/', 
         'WarpDoc/digital/random/',
         200,
         "Random folder",
         scale_factor=0.3, 
         num_images=50
     )
print("----------------------- SSIM calculation with ", 200," ransac distance threshold is finished -----------------------")


# In[52]:


print("------------------------------------------------------------------------------")
print(f"\nAverage SSIM of the Random folder with a RANSAC distance threshold of 200 : {average_ssim_random:.4f}")


# In[ ]:





# In[ ]:





# # COMPUTE AVERAGE SSIM FOR ROTATE FOLDER

# ### Ransac threshold : 125

# In[110]:


full_results_rotate, average_ssim_rotate = calculate_results_and_ssim_for_folder(
         'WarpDoc/distorted/rotate/', 
         'WarpDoc/digital/rotate/',
         125,
         "Random folder",
         scale_factor=0.25, 
         num_images=50
     )
print("----------------------- SSIM calculation with ", 125," ransac distance threshold is finished -----------------------")


# In[111]:


print("------------------------------------------------------------------------------")
print(f"\nAverage SSIM of the Rotate folder with a RANSAC distance threshold of 125 : {average_ssim_rotate:.4f}")



# ### Ransac threshold : 150

# In[112]:


full_results_rotate, average_ssim_rotate = calculate_results_and_ssim_for_folder(
         'WarpDoc/distorted/rotate/', 
         'WarpDoc/digital/rotate/',
         150,
         "Rotate folder",
         scale_factor=0.25, 
         num_images=50
     )
print("----------------------- SSIM calculation with ", 150," ransac distance threshold is finished -----------------------")


# In[113]:


print("------------------------------------------------------------------------------")
print(f"\nAverage SSIM of the Rotate folder with a RANSAC distance threshold of 150 : {average_ssim_rotate:.4f}")


# ### Ransac threshold : 200

# In[114]:


full_results_rotate, average_ssim_rotate = calculate_results_and_ssim_for_folder(
         'WarpDoc/distorted/rotate/', 
         'WarpDoc/digital/rotate/',
         200,
         "Rotate folder",
         scale_factor=0.3, 
         num_images=50
     )
print("----------------------- SSIM calculation with ", 200," ransac distance threshold is finished -----------------------")


# In[115]:


print("------------------------------------------------------------------------------")
print(f"\nAverage SSIM of the Rotate folder with a RANSAC distance threshold of 200 : {average_ssim_rotate:.4f}")


# # -----------------------------------------------
# # 10. Overall Results
# # -----------------------------------------------

# In[119]:


folders = ["Curved", "Fold", "Incomplete", "Perspective", "Random", "Rotate"]
ransac_thresholds = ["125", "150", "200"]
ssim_values = [
    [0.5345, 0.5260, 0.4922],  # Curved
    [0.4895, 0.5164, 0.4963],  # Fold
    [0.4293, 0.4285, 0.4511],  # Incomplete
    [0.5230, 0.5435, 0.5119],  # Perspective
    [0.5086, 0.5132, 0.5078],  # Random
    [0.4576, 0.4453, 0.4663],  # Rotate
]


colors = ["#FF6347", "#4682B4", "#32CD32", "#FFD700", "#9370DB", "#FF69B4"]


fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.2  
x = np.arange(len(folders))

for i, threshold in enumerate(ransac_thresholds):
    bar_positions = x + i * bar_width  
    bars = ax.bar(bar_positions, [ssim[i] for ssim in ssim_values], 
                  bar_width, label=f"RANSAC {threshold}", color=colors[i % len(colors)])
    

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height / 2, 
                f"{height:.4f}", ha='center', va='center', rotation=90, color="white", fontsize=10)


ax.set_xlabel("Folders")
ax.set_ylabel("Average SSIM")
ax.set_title("Average SSIM Values for Different RANSAC Thresholds")
ax.set_xticks(x + bar_width)
ax.set_xticklabels(folders)
ax.legend(title="RANSAC Threshold")

plt.tight_layout()
plt.show()


# In[ ]:




