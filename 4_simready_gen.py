"""
===============================================================================
4_simready_gen.py - Simulation-Ready Asset Generation (URDF & MJCF)
===============================================================================

This script generates physics simulation-ready assets from VLM outputs:
    - URDF (Unified Robot Description Format) for ROS/PyBullet
    - MJCF (MuJoCo-format XML) for MuJoCo physics simulation

Pipeline Overview:
    1. Parse VLM output to extract:
       - Object name, category, dimensions
       - Part information (materials, physical properties)
       - Kinematic groups (fixed, sliding, revolute joints)
    2. Generate URDF with proper joint hierarchy
    3. Generate MJCF with physics parameters, textures, and materials

Joint Types Supported:
    - A (Free): Floating joint (6-DOF)
    - B (Slide): Prismatic joint (1-DOF translation)
    - C (Revolute): Hinge joint (1-DOF rotation)
    - D (Ball): Ball-and-socket joint (3-DOF rotation)
    - CB (Combined): Revolute + Slide joint

Key Concepts:
    - Voxel Grid: 32x32x32 grid for position calculation
    - Group Info: Defines kinematic relationships between parts
    - Joint Parameters: Direction vectors, positions, and motion ranges

Dependencies:
    - trimesh: Mesh loading for volume calculation
    - xml.etree: XML generation for URDF/MJCF

Author: PhysX-Anything Team
===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import re
import json
import shutil
import logging
import argparse
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from collections import defaultdict, deque

import numpy as np
import trimesh
from scipy.spatial import cKDTree as KDTree

# Debugging (can be removed in production)
import ipdb


# =============================================================================
# LOGGING SETUP
# =============================================================================

def get_logger(filename, verbosity=1, name=None):
    """
    Create a logger that writes to both file and console.
    
    Args:
        filename (str): Log file path
        verbosity (int): 0=DEBUG, 1=INFO, 2=WARNING
        name (str): Logger name (optional)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# =============================================================================
# ADJACENT REGION DETECTION
# =============================================================================

def _pairwise_nn(a: np.ndarray, b: np.ndarray):
    """
    Compute nearest neighbor correspondences between two point clouds.
    
    Uses KD-trees for efficient nearest neighbor queries in both directions
    (A->B and B->A) to find mutual nearest neighbors.
    
    Args:
        a (np.ndarray): First point cloud, shape (N, 3)
        b (np.ndarray): Second point cloud, shape (M, 3)
    
    Returns:
        tuple: (idx_ab, dist_ab, idx_ba, dist_ba) where:
            - idx_ab: For each point in A, index of nearest point in B
            - dist_ab: Distance to nearest point in B
            - idx_ba: For each point in B, index of nearest point in A
            - dist_ba: Distance to nearest point in A
    """
    ta = KDTree(a)
    tb = KDTree(b)
    dist_ab, idx_ab = tb.query(a, k=1, workers=-1)
    dist_ba, idx_ba = ta.query(b, k=1, workers=-1)
    
    return idx_ab, dist_ab, idx_ba, dist_ba


def _robust_threshold(d, method="mad", q=0.2, k=2.5):
    """
    Compute a robust threshold for outlier detection.
    
    Supports two methods:
        - "quantile": Simple quantile threshold
        - "mad": Median Absolute Deviation (more robust to outliers)
    
    Args:
        d (np.ndarray): Distance values
        method (str): "quantile" or "mad"
        q (float): Quantile for quantile method (default 0.2)
        k (float): MAD multiplier (default 2.5)
    
    Returns:
        float: Computed threshold value
    """
    d = np.asarray(d)
    if method == "quantile":
        return np.quantile(d, q)
    
    # MAD (Median Absolute Deviation) method
    med = np.median(d)
    mad = np.median(np.abs(d - med)) + 1e-12
    return med + k * 1.4826 * mad  # 1.4826 is scale factor for normal distribution


def find_adjacent_region(
    a: np.ndarray,
    b: np.ndarray,
    thr: float | None = None,
    thr_mode: str = "mad",
    q: float = 0.2,
    expand_radius: float | None = None,
):
    """
    Find the adjacent/contact region between two point clouds.
    
    This function identifies points that are close to each other in two point clouds,
    useful for determining where two parts of an object meet (e.g., joint locations).
    
    Algorithm:
        1. Find mutual nearest neighbors between A and B
        2. Filter by distance threshold (auto-computed or provided)
        3. Optionally expand the region by radius
        4. Fit a plane to the midpoints (for joint axis estimation)
    
    Args:
        a (np.ndarray): First point cloud, shape (N, 3)
        b (np.ndarray): Second point cloud, shape (M, 3)
        thr (float): Distance threshold (None = auto-compute)
        thr_mode (str): Threshold computation method ("mad" or "quantile")
        q (float): Quantile for threshold computation
        expand_radius (float): Expand region by this radius
    
    Returns:
        dict: Contains:
            - a_idx: Indices of adjacent points in A
            - b_idx: Indices of adjacent points in B
            - pairs: (N, 2) array of corresponding point pairs
            - midpoints: (N, 3) array of midpoints between pairs
            - plane: (center, normal) tuple or None
            - thr: Threshold used
    """
    assert a.ndim == 2 and a.shape[1] == 3
    assert b.ndim == 2 and b.shape[1] == 3

    idx_ab, dist_ab, idx_ba, dist_ba = _pairwise_nn(a, b)

    # Find mutual nearest neighbors (A->B->A should return to same point)
    mutual = np.arange(len(a)) == idx_ba[idx_ab]
    d_mutual = dist_ab[mutual]
    i_a = np.nonzero(mutual)[0]
    j_b = idx_ab[mutual]

    # Handle empty case
    if len(i_a) == 0:
        return dict(
            a_idx=np.array([], dtype=int),
            b_idx=np.array([], dtype=int),
            pairs=np.empty((0, 2), dtype=int),
            midpoints=np.empty((0, 3), dtype=a.dtype),
            plane=None,
            thr=0.0
        )

    # Compute threshold
    used_thr = _robust_threshold(d_mutual, thr_mode, q=q) if thr is None else thr

    # Filter by threshold
    keep = d_mutual <= used_thr
    i_a = i_a[keep]
    j_b = j_b[keep]
    d_kept = d_mutual[keep]

    # Fallback to looser threshold if nothing passes
    if len(i_a) == 0 and len(d_mutual) > 0 and thr is None:
        used_thr = _robust_threshold(d_mutual, "quantile", q=max(0.4, q))
        keep = d_mutual <= used_thr
        i_a = np.nonzero(mutual)[0][keep]
        j_b = idx_ab[mutual][keep]

    # Build output arrays
    pairs = np.stack([i_a, j_b], axis=1) if len(i_a) else np.empty((0, 2), dtype=int)
    midpoints = (a[i_a] + b[j_b]) * 0.5 if len(i_a) else np.empty((0, 3), dtype=a.dtype)

    # Helper to expand region within a point cloud
    def _expand_within_cloud(points, seeds, radius):
        if radius is None or len(seeds) == 0:
            return np.unique(seeds)
        
        t = KDTree(points)
        idxs = set(seeds.tolist())
        for p in points[seeds]:
            hits = t.query_ball_point(p, r=radius)
            idxs.update(hits)
        return np.fromiter(idxs, dtype=int)

    # Expand and deduplicate indices
    a_idx = np.unique(i_a)
    b_idx = np.unique(j_b)
    a_idx = _expand_within_cloud(a, a_idx, expand_radius)
    b_idx = _expand_within_cloud(b, b_idx, expand_radius)

    # Fit plane to midpoints (for joint axis estimation)
    plane = None
    if len(midpoints) >= 3:
        c = midpoints.mean(axis=0)
        X = midpoints - c
        # SVD to find normal (smallest singular value direction)
        _, _, vh = np.linalg.svd(X, full_matrices=False)
        n = vh[-1]  # Normal is last row of Vh
        n = n / (np.linalg.norm(n) + 1e-12)
        plane = (c, n)

    return dict(
        a_idx=a_idx,
        b_idx=b_idx,
        pairs=pairs,
        midpoints=midpoints,
        plane=plane,
        thr=float(used_thr),
    )


# =============================================================================
# VOXEL GRID OPERATIONS
# =============================================================================

# Grid configuration
GRID = 32  # Voxel grid resolution
NEI6 = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1]
], dtype=np.int8)  # 6-connectivity neighborhood offsets


def rasterize(points, grid=GRID):
    """
    Convert point coordinates to a binary occupancy grid.
    
    Args:
        points (np.ndarray): Point coordinates, shape (N, 3)
        grid (int): Grid resolution
    
    Returns:
        np.ndarray: Binary occupancy grid, shape (grid, grid, grid)
    """
    occ = np.zeros((grid, grid, grid), dtype=bool)
    pts = np.asarray(points, dtype=np.int16)
    
    # Filter out-of-bounds points
    mask = ((pts >= 0) & (pts < grid)).all(1)
    x, y, z = pts[mask].T
    occ[x, y, z] = True
    
    return occ


def boundary_mask(occ):
    """
    Find boundary voxels (occupied voxels adjacent to empty space).
    
    Args:
        occ (np.ndarray): Binary occupancy grid
    
    Returns:
        np.ndarray: Binary mask of boundary voxels
    """
    bnd = np.zeros_like(occ)
    xs, ys, zs = np.where(occ)
    
    # Check 6-connectivity neighbors
    for dx, dy, dz in NEI6:
        x2 = np.clip(xs + dx, 0, occ.shape[0] - 1)
        y2 = np.clip(ys + dy, 0, occ.shape[1] - 1)
        z2 = np.clip(zs + dz, 0, occ.shape[2] - 1)
        # Mark as boundary if neighbor is empty
        bnd[xs, ys, zs] |= ~occ[x2, y2, z2]
    
    return bnd


def idx_to_xyz(idx):
    """
    Convert a boolean mask to XYZ coordinates.
    
    Args:
        idx (np.ndarray): Boolean mask
    
    Returns:
        np.ndarray: Coordinates of True values, shape (N, 3)
    """
    return np.stack(np.where(idx), axis=1).astype(np.int16)


def most_adjacent_shell_6n(A_xyz, B_xyz, grid=GRID):
    """
    Find the contact region between two voxel regions using wave propagation.
    
    This function determines where two parts of an object meet by:
        1. Finding boundary voxels of each region
        2. Checking for direct 6-connectivity contact
        3. If no contact, propagating waves until they meet
    
    This is useful for determining joint positions between parts.
    
    Args:
        A_xyz (np.ndarray): Voxel coordinates of region A, shape (N, 3)
        B_xyz (np.ndarray): Voxel coordinates of region B, shape (M, 3)
        grid (int): Grid resolution
    
    Returns:
        dict: Contains:
            - metric: "6-neighbor steps"
            - min_grid_distance: Number of grid steps between regions
            - pairs: Array of touching voxel pairs
            - midpoints: Midpoints between touching pairs
    """
    # Rasterize point clouds to occupancy grids
    A = rasterize(A_xyz, grid)
    B = rasterize(B_xyz, grid)
    
    # Find boundary voxels (surface of each region)
    A_front = boundary_mask(A) & A
    B_front = boundary_mask(B) & B

    # Check for direct contact (6-connectivity)
    touch_pairs = []
    if A_front.any() and B_front.any():
        Ax, Ay, Az = np.where(A_front)
        Aset = set(zip(Ax, Ay, Az))
        B_occ = B
        
        # Check each neighbor direction
        for dx, dy, dz in NEI6:
            nb = (
                np.clip(Ax + dx, 0, grid - 1),
                np.clip(Ay + dy, 0, grid - 1),
                np.clip(Az + dz, 0, grid - 1)
            )
            hit = B_occ[nb]
            if hit.any():
                for (x, y, z), (x2, y2, z2), h in zip(
                    zip(Ax, Ay, Az), zip(*nb), hit
                ):
                    if h:
                        touch_pairs.append(((x, y, z), (x2, y2, z2)))
        
        # If direct contact found, return immediately
        if touch_pairs:
            mid = np.array([
                (np.array(a) + np.array(b)) / 2.0 
                for a, b in touch_pairs
            ], dtype=np.float32)
            return {
                "metric": "6-neighbor steps",
                "min_grid_distance": 1,
                "pairs": np.array(touch_pairs, dtype=np.int16),
                "midpoints": mid,
            }

    # No direct contact - use wave propagation to find meeting point
    A_wave = A_front.copy()
    B_wave = B_front.copy()
    visitedA = A_front.copy()
    visitedB = B_front.copy()
    dist = 1  # Distance counter

    while True:
        def dilate_once(wave, solid):
            """Expand wave by one voxel in all 6 directions."""
            xs, ys, zs = np.where(wave)
            nxt = np.zeros_like(wave)
            for dx, dy, dz in NEI6:
                x2 = np.clip(xs + dx, 0, grid - 1)
                y2 = np.clip(ys + dy, 0, grid - 1)
                z2 = np.clip(zs + dz, 0, grid - 1)
                nxt[x2, y2, z2] = True
            nxt &= ~solid  # Don't expand into solid region
            return nxt

        # Expand both waves
        A_next = dilate_once(A_wave, A)
        B_next = dilate_once(B_wave, B)

        # Remove already-visited voxels
        A_next &= ~visitedA
        B_next &= ~visitedB

        visitedA |= A_next
        visitedB |= B_next

        # Check if waves meet
        meet = A_next & visitedB
        if meet.any():
            meet_xyz = idx_to_xyz(meet)
            
            # Find contact pairs
            B_prev = B_wave
            pairs = []
            for x, y, z in meet_xyz:
                for dx, dy, dz in NEI6:
                    x2 = np.clip(x + dx, 0, grid - 1)
                    y2 = np.clip(y + dy, 0, grid - 1)
                    z2 = np.clip(z + dz, 0, grid - 1)
                    if B_prev[x2, y2, z2]:
                        pairs.append(((x, y, z), (x2, y2, z2)))
            
            pairs = np.unique(np.array(pairs, dtype=np.int16), axis=0)
            mid = (pairs[:, 0, :].astype(np.float32) + pairs[:, 1, :].astype(np.float32)) / 2.0
            
            return {
                "metric": "6-neighbor steps",
                "min_grid_distance": dist + 1,
                "pairs": pairs,
                "midpoints": mid,
            }

        # Check if no more expansion possible (disconnected regions)
        if not (A_next.any() or B_next.any()):
            return {
                "metric": "6-neighbor steps",
                "min_grid_distance": None,
                "pairs": np.zeros((0, 2, 3), np.int16),
                "midpoints": np.zeros((0, 3), np.float32)
            }
        
        A_wave, B_wave = A_next, B_next
        dist += 1


# =============================================================================
# BOUNDING BOX UTILITIES
# =============================================================================

def bbox_corners_and_edge_midpoints(pts: np.ndarray):
    """
    Compute bounding box corners, edge midpoints, and center for a point cloud.
    
    Useful for determining potential joint attachment points.
    
    Args:
        pts (np.ndarray): Point cloud, shape (N, 3)
    
    Returns:
        tuple: (corners, edge_mids, center) where:
            - corners: 8 corner points of bounding box
            - edge_mids: 12 edge midpoints
            - center: Center point of bounding box
    """
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    x0, y0, z0 = mins
    x1, y1, z1 = maxs

    # Generate 8 corners
    corners = np.array([
        [x, y, z]
        for x in [x0, x1]
        for y in [y0, y1]
        for z in [z0, z1]
    ], dtype=float)
    corners = np.unique(corners, axis=0)

    # Generate 12 edge midpoints
    xm, ym, zm = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    edge_mids = []
    
    # X-parallel edges (4 edges)
    for y in [y0, y1]:
        for z in [z0, z1]:
            edge_mids.append([xm, y, z])
    
    # Y-parallel edges (4 edges)
    for x in [x0, x1]:
        for z in [z0, z1]:
            edge_mids.append([x, ym, z])
    
    # Z-parallel edges (4 edges)
    for x in [x0, x1]:
        for y in [y0, y1]:
            edge_mids.append([x, y, zm])
    
    edge_mids = np.unique(np.array(edge_mids, dtype=float), axis=0)

    center = np.array([xm, ym, zm], dtype=float)
    
    return corners, edge_mids, center


# =============================================================================
# JOINT POSITION CANDIDATE GENERATION
# =============================================================================

def generate_allcandidate(ind_a_index, ind_b_index, datapath):
    """
    Generate candidate joint positions between two groups of parts.
    
    Finds the contact region between two groups and returns the centers
    of each group's contact area as potential joint positions.
    
    Args:
        ind_a_index (list): Part indices for group A
        ind_b_index (list): Part indices for group B
        datapath (str): Path containing ind_{i}.npy files
    
    Returns:
        np.ndarray: Candidate positions, shape (2, 3), normalized to [-0.5, 0.5]
    """
    # Load voxel coordinates for both groups
    ind_a = []
    ind_b = []
    
    for ind in ind_a_index:
        ind_a.append(np.load(os.path.join(datapath, f'ind_{ind}.npy')))
    
    for ind in ind_b_index:
        ind_b.append(np.load(os.path.join(datapath, f'ind_{ind}.npy')))

    ind_a = np.concatenate(ind_a)
    ind_b = np.concatenate(ind_b)

    # Find adjacent region using wave propagation
    results = most_adjacent_shell_6n(ind_a, ind_b)
    ind_a_nei = results['pairs'][:, 0]
    ind_b_nei = results['pairs'][:, 1]

    # Get center of each contact region
    corners, edge_mids, center = bbox_corners_and_edge_midpoints(ind_a_nei)
    bbox_corners_a = np.concatenate([center[None]])  # Just use center
    
    corners, edge_mids, center = bbox_corners_and_edge_midpoints(ind_b_nei)
    bbox_corners_b = np.concatenate([center[None]])  # Just use center

    # Combine and normalize to [-0.5, 0.5] range
    allcandidate = np.concatenate([bbox_corners_a, bbox_corners_b])
    allcandidate = allcandidate / 32 - 0.5
    
    return allcandidate


def generate_allcandidate_center(ind_a_index, ind_b_index, datapath):
    """
    Generate a single joint position candidate at the midpoint of contact regions.
    
    Similar to generate_allcandidate but returns the average of all candidate points.
    
    Args:
        ind_a_index (list): Part indices for group A
        ind_b_index (list): Part indices for group B
        datapath (str): Path containing ind_{i}.npy files
    
    Returns:
        np.ndarray: Single candidate position, shape (3,), normalized to [-0.5, 0.5]
    """
    # Load voxel coordinates for both groups
    ind_a = []
    ind_b = []
    
    for ind in ind_a_index:
        ind_a.append(np.load(os.path.join(datapath, f'ind_{ind}.npy')))
    
    for ind in ind_b_index:
        ind_b.append(np.load(os.path.join(datapath, f'ind_{ind}.npy')))

    ind_a = np.concatenate(ind_a)
    ind_b = np.concatenate(ind_b)

    # Find adjacent region
    results = most_adjacent_shell_6n(ind_a, ind_b)
    ind_a_nei = results['pairs'][:, 0]
    ind_b_nei = results['pairs'][:, 1]

    # Get mean of all candidate points from each side
    corners, edge_mids, center = bbox_corners_and_edge_midpoints(ind_a_nei)
    bbox_corners_a = np.concatenate([corners, edge_mids, center[None]]).mean(0)
    
    corners, edge_mids, center = bbox_corners_and_edge_midpoints(ind_b_nei)
    bbox_corners_b = np.concatenate([corners, edge_mids, center[None]]).mean(0)

    # Return midpoint between the two centers
    allcandidate = (bbox_corners_a + bbox_corners_b) / 2
    allcandidate = allcandidate / 32 - 0.5
    
    return allcandidate


# =============================================================================
# URDF XML GENERATION UTILITIES
# =============================================================================

def make_origin_element(xyz, rpy):
    """
    Create an XML <origin> element with position and rotation.
    
    Args:
        xyz (list): Position as list of strings ["x", "y", "z"]
        rpy (list): Rotation (roll, pitch, yaw) as list of strings
    
    Returns:
        ET.Element: XML origin element
    """
    origin = ET.Element('origin')
    origin.set('xyz', ' '.join(xyz))
    origin.set('rpy', ' '.join(rpy))
    return origin


def add_inertial(link_element, xyz="0 0 0"):
    """
    Add default inertial properties to a URDF link.
    
    Creates <inertial> element with default mass and inertia values.
    
    Args:
        link_element (ET.Element): Parent link element
        xyz (str): Center of mass position
    """
    inertial = ET.SubElement(link_element, 'inertial')
    ET.SubElement(inertial, 'origin', xyz=xyz, rpy="0 0 0")
    ET.SubElement(inertial, 'mass', value="1.0")
    ET.SubElement(
        inertial, 'inertia',
        ixx="1.0", ixy="0.0", ixz="0.0",
        iyy="1.0", iyz="0.0", izz="1.0"
    )


def add_fixed_joint(robot, name, parent, child, xyz="0 0 0", rpy="0 0 0"):
    """
    Add a fixed joint between two links in a URDF.
    
    Args:
        robot (ET.Element): Root robot element
        name (str): Joint name
        parent (str): Parent link name
        child (str): Child link name
        xyz (str): Joint position offset
        rpy (str): Joint rotation offset
    
    Returns:
        ET.Element: The created joint element
    """
    joint = ET.SubElement(robot, "joint", name=name, type="fixed")
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    ET.SubElement(joint, "origin", xyz=xyz, rpy=rpy)
    return joint


# =============================================================================
# TEXT PARSING UTILITIES
# =============================================================================

def _to_nums(lst, expect_len):
    """
    Convert a list of strings to floats with padding/truncation.
    
    Args:
        lst (list): List of string values
        expect_len (int): Expected output length
    
    Returns:
        list: List of floats with exactly expect_len elements
    """
    out = []
    for s in lst:
        s = s.strip()
        if s:
            try:
                v = float(int(s))
            except ValueError:
                try:
                    v = float(s)
                except ValueError:
                    v = 0.0
        else:
            v = 0.0
        out.append(v)
    
    # Pad or truncate to expected length
    if len(out) < expect_len:
        out += [0.0] * (expect_len - len(out))
    elif len(out) > expect_len:
        out = out[:expect_len]
    
    return out


def clean_npfloat64(values):
    """
    Clean numpy float64 string representations from VLM output.
    
    The VLM sometimes outputs values like "np.float64(0.5)" which need
    to be extracted as plain numbers.
    
    Args:
        values (list): List of strings possibly containing np.float64()
    
    Returns:
        list: Cleaned string values
    """
    cleaned = []
    for s in values:
        s = s.strip()
        if s.startswith('np.float64('):
            # Extract number from np.float64(X)
            num_str = re.sub(r'.*?\((.*?)\)', r'\1', s)
            cleaned.append(num_str)
        else:
            cleaned.append(s)
    return cleaned


def _extract_bracket_list(block, key, expect_len):
    """
    Extract a bracketed list of numbers from text.
    
    Searches for patterns like "key: [1, 2, 3]" and extracts the values.
    
    Args:
        block (str): Text block to search
        key (str): Key name to look for
        expect_len (int): Expected number of values
    
    Returns:
        list: Extracted float values, padded to expect_len
    """
    pattern = rf'{re.escape(key)}[^:\[]*:\s*\[([^\]]*)\]'
    m = re.search(pattern, block, flags=re.IGNORECASE)
    
    if not m:
        return [0.0] * expect_len
    
    raw = m.group(1)
    items = [x for x in raw.split(',')]
    items = clean_npfloat64(items)
    
    return _to_nums(items, expect_len)


# =============================================================================
# MJCF XML TREE MANIPULATION
# =============================================================================

def find_body_by_name(root: ET.Element, name: str) -> ET.Element:
    """
    Find a body element by name in an MJCF tree.
    
    Args:
        root (ET.Element): Root element to search
        name (str): Body name to find
    
    Returns:
        ET.Element: Found body element or None
    """
    for elem in root.iter("body"):
        if elem.get("name") == name:
            return elem
    return None


def move_element(child: ET.Element, new_parent: ET.Element):
    """
    Move an XML element from its current parent to a new parent.
    
    Note: This is a helper function for XML tree manipulation.
    
    Args:
        child (ET.Element): Element to move
        new_parent (ET.Element): New parent element
    """
    old_parent = child.getparent() if hasattr(child, "getparent") else None
    
    if old_parent is None:
        for elem in new_parent.iter():
            pass
    
    def _find_parent(root, node):
        for e in root.iter():
            for c in list(e):
                if c is node:
                    return e
        return None

    root = new_parent
    while root.getparent() is not None if hasattr(root, "getparent") else False:
        root = root.getparent()

    parent = _find_parent(root, child)
    if parent is not None:
        parent.remove(child)
    new_parent.append(child)


def reparent_by_group_info(
    mjcf_root: ET.Element, 
    group_info: dict,
    base_body_name: str = "base",
    group_body_prefix: str = "grouppart_"
):
    """
    Reparent body elements in MJCF tree based on kinematic group hierarchy.
    
    This function restructures the MJCF XML tree so that child groups are
    properly nested under their parent groups, creating the correct kinematic chain.
    
    Args:
        mjcf_root (ET.Element): Root of MJCF document
        group_info (dict): Group hierarchy information from VLM output
            Format: {group_id: [members, parent_group, params, type]}
        base_body_name (str): Name of the base body
        group_body_prefix (str): Prefix for group body names
    """
    # Build parent-child relationships
    parent_of = {}
    for gkey, gval in group_info.items():
        if str(gkey) == "0":
            continue  # Skip base group
        
        try:
            parent_str = str(gval[1])
        except Exception as e:
            raise ValueError(f"group_info['{gkey}'] lacks parent group: {gval}") from e
        parent_of[str(gkey)] = parent_str

    # Find base body
    base_body = find_body_by_name(mjcf_root, base_body_name)
    if base_body is None:
        raise ValueError(f"Cannot find base body: name='{base_body_name}'")

    def body_name_for_group(gid: str) -> str:
        """Get body name for a group ID."""
        if gid == "0":
            return base_body_name
        return f"{group_body_prefix}{gid}"

    # Build dependency graph for topological sort
    children_of = defaultdict(list)
    indeg = defaultdict(int)
    nodes = set(["0"])  # Include base group
    
    for c, p in parent_of.items():
        nodes.add(c)
        nodes.add(p)
        children_of[p].append(c)
        indeg[c] += 1
        indeg.setdefault(p, 0)

    # Topological sort to process parents before children
    q = deque([n for n in nodes if indeg[n] == 0])
    topo = []
    
    while q:
        u = q.popleft()
        topo.append(u)
        for v in children_of[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(topo) != len(nodes):
        raise ValueError("Detected cycle in group_info hierarchy")

    # Reparent bodies in topological order
    for gid in topo:
        if gid == "0":
            continue
        
        child_name = body_name_for_group(gid)
        parent_name = body_name_for_group(parent_of[gid])

        child_body = find_body_by_name(mjcf_root, child_name)
        parent_body = find_body_by_name(mjcf_root, parent_name)

        if child_body is None:
            print(f"Skipping: {child_name} (not found)")
            continue
        
        if parent_body is None:
            raise ValueError(
                f"Cannot find parent body: {parent_name} "
                f"(child group: {gid}, parent group: {parent_of[gid]})"
            )

        # Check if already a child
        already_child = False
        for c in list(parent_body):
            if c is child_body:
                already_child = True
                break
        
        if already_child:
            continue

        # Find current parent and reparent
        def find_parent(root, node):
            for e in mjcf_root.iter():
                for c in list(e):
                    if c is node:
                        return e
            return None

        old_parent = find_parent(mjcf_root, child_body)
        if old_parent is not None:
            old_parent.remove(child_body)
        parent_body.append(child_body)


def _indent(elem, level=0):
    """
    Add indentation to XML element for pretty printing.
    
    Args:
        elem (ET.Element): Element to indent
        level (int): Current indentation level
    """
    i = "\n" + level * "    "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "    "
        for e in elem:
            _indent(e, level + 1)
        if not e.tail or not e.tail.strip():
            e.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


# =============================================================================
# MJCF GENERATION
# =============================================================================

def generate_mjcf(
    jsondata: dict = {},
    fixed_base: int = 0,
    out_path: str = "test.xml",
    model_name: str = "test",
    # Physics / simulation options
    angle_unit: str = "radian",
    timestep: float = 0.002,
    gravity: str = "0 0 -9.81",
    wind: str = "0 0 0",
    integrator: str = "implicitfast",
    density: float = 1.225,
    viscosity: float = 1.8e-5,
    # Visual settings
    realtime: int = 1,
    shadowsize: int = 16384,
    numslices: int = 28,
    offsamples: int = 4,
    headlight_diffuse: str = "2 2 2",
    headlight_specular: str = "0.5 0.5 0.5",
    headlight_active: int = 1,
    rgba_fog: str = "0 1 0 1",
    rgba_haze: str = "1 0 0 1",
    # Skybox
    skybox_file: Optional[str] = "./desert.png",
    skybox_gridsize: str = "3 4",
    skybox_gridlayout: str = ".U..LFRB.D..",
    # Ground plane texture
    plane_texture_name: str = "plane",
    plane_material_name: str = "plane",
    plane_rgb1: str = ".1 .1 .1",
    plane_rgb2: str = ".5 .5 .5",
    plane_width: int = 512,
    plane_height: int = 512,
    plane_mark: str = "cross",
    plane_markrgb: str = ".8 .8 .8",
    plane_reflectance: float = 0.3,
    plane_texrepeat: str = "1 1",
    plane_texuniform: str = "true",
    # Contact / physics defaults
    geom_solref: str = ".5e-4",
    geom_solimp: str = "0.9 0.99 1e-4",
    geom_fluidcoef: str = "0.5 0.25 0.5 2.0 1.0",
    # Part definitions
    parts: List[Dict] = None,
    # World layout
    floor_condim: int = 6,
    floor_size: str = "0 0 .25",
    light_pos: str = "30 30 30",
    light_dir: str = "0 -2 -1",
    light_ambient: str = ".3 .3 .3",
    light_diffuse: str = ".5 .5 .5",
    light_specular: str = ".5 .5 .5",
    # Object placement
    base_pos: str = "0 0 1",
    base_euler: str = "0 0 0",
    part_pos: str = "0 0 1.2",
    part_euler: str = "1.5 0 0",
    deformable: int = 0,
):
    """
    Generate a complete MJCF (MuJoCo XML) file for physics simulation.
    
    This function creates a fully configured MJCF file including:
        - Physics simulation parameters
        - Visual rendering settings
        - Asset definitions (meshes, textures, materials)
        - World layout (ground, lights)
        - Object bodies with joints based on group_info
    
    Joint Types (from group_info[-1]):
        - 'A': Free joint (6-DOF floating)
        - 'B': Slide/Prismatic joint (1-DOF translation)
        - 'C': Revolute/Hinge joint (1-DOF rotation)
        - 'D': Ball joint (3-DOF rotation)
        - 'CB': Combined revolute + slide joint
    
    Args:
        jsondata (dict): Parsed VLM output containing group_info, parts, etc.
        fixed_base (int): Whether to fix the base (0=free, 1=fixed)
        out_path (str): Output file path
        model_name (str): Model name in MJCF
        ... (many physics and visual parameters)
        parts (List[Dict]): Part configurations with mesh/texture paths
        deformable (int): Whether to use deformable objects (0=rigid, 1=flex)
    
    Returns:
        str: Output file path
    """
    if parts is None or len(parts) == 0:
        raise ValueError("At least one part must be provided")

    # =========================================================================
    # Create root MJCF structure
    # =========================================================================
    mujoco = ET.Element("mujoco", attrib={"model": model_name})
    
    # Compiler settings
    ET.SubElement(mujoco, "compiler", attrib={
        "angle": angle_unit, 
        "autolimits": "true"
    })
    
    # Simulation options
    ET.SubElement(mujoco, "option", attrib={
        "timestep": f"{timestep}",
        "gravity": gravity,
        "wind": wind,
        "integrator": integrator,
        "density": f"{density}",
        "viscosity": f"{viscosity}",
    })

    # =========================================================================
    # Visual settings
    # =========================================================================
    visual = ET.SubElement(mujoco, "visual")
    ET.SubElement(visual, "global", attrib={"realtime": str(realtime)})
    ET.SubElement(visual, "quality", attrib={
        "shadowsize": str(shadowsize),
        "numslices": str(numslices),
        "offsamples": str(offsamples)
    })
    ET.SubElement(visual, "headlight", attrib={
        "diffuse": headlight_diffuse,
        "specular": headlight_specular,
        "active": str(headlight_active)
    })
    ET.SubElement(visual, "rgba", attrib={
        "fog": rgba_fog,
        "haze": rgba_haze
    })

    # =========================================================================
    # Assets (meshes, textures, materials)
    # =========================================================================
    asset = ET.SubElement(mujoco, "asset")

    # Add mesh, texture, and material for each part
    for p in parts:
        pname = p["name"]
        
        # Mesh asset
        ET.SubElement(asset, "mesh", attrib={
            "name": pname,
            "file": p["mesh_file"],
            "scale": p.get("scale", "1 1 1")
        })
        
        # Texture asset
        tex_name = f"{pname}_tex"
        ET.SubElement(asset, "texture", attrib={
            "type": "2d",
            "name": tex_name,
            "file": p["tex_file"]
        })
        
        # Material asset
        mat_name = f"{pname}_img"
        ET.SubElement(asset, "material", attrib={
            "name": mat_name,
            "texture": tex_name
        })

    # Skybox texture
    if skybox_file:
        ET.SubElement(asset, "texture", attrib={
            "type": "skybox",
            "file": skybox_file,
            "gridsize": skybox_gridsize,
            "gridlayout": skybox_gridlayout
        })

    # Ground plane texture and material
    ET.SubElement(asset, "texture", attrib={
        "name": plane_texture_name,
        "type": "2d",
        "builtin": "checker",
        "rgb1": plane_rgb1,
        "rgb2": plane_rgb2,
        "width": str(plane_width),
        "height": str(plane_height),
        "mark": plane_mark,
        "markrgb": plane_markrgb,
    })
    ET.SubElement(asset, "material", attrib={
        "name": plane_material_name,
        "reflectance": str(plane_reflectance),
        "texture": plane_texture_name,
        "texrepeat": plane_texrepeat,
        "texuniform": plane_texuniform,
    })

    # =========================================================================
    # Default settings
    # =========================================================================
    default = ET.SubElement(mujoco, "default")
    ET.SubElement(default, "geom", attrib={
        "solref": geom_solref,
        "solimp": geom_solimp,
        "fluidcoef": geom_fluidcoef
    })

    # Per-part default classes
    for p in parts:
        pname = p["name"]
        dclass = ET.SubElement(default, "default", attrib={"class": pname})
        
        attrib = {
            "type": "mesh",
            "mesh": pname,
            "contype": p.get("contype", "1"),
            "conaffinity": p.get("conaffinity", "1"),
        }
        
        if "density" in p:
            attrib["density"] = str(p["density"])
        if "fluidshape" in p:
            attrib["fluidshape"] = p["fluidshape"]
        
        ET.SubElement(dclass, "geom", attrib=attrib)

    # =========================================================================
    # World body (floor, lights, objects)
    # =========================================================================
    world = ET.SubElement(mujoco, "worldbody")
    
    # Ground plane
    ET.SubElement(world, "geom", attrib={
        "name": "floor",
        "pos": "0 0 0",
        "size": floor_size,
        "type": "plane",
        "material": plane_material_name,
        "condim": str(floor_condim),
    })
    
    # Directional light
    ET.SubElement(world, "light", attrib={
        "directional": "true",
        "ambient": light_ambient,
        "pos": light_pos,
        "dir": light_dir,
        "diffuse": light_diffuse,
        "specular": light_specular,
    })

    # =========================================================================
    # Base body (root of the object)
    # =========================================================================
    base_body = ET.SubElement(world, "body", attrib={
        "name": "base",
        "pos": base_pos,
        "euler": base_euler
    })
    
    # Add free joint if base is not fixed
    if fixed_base == 0:
        ET.SubElement(base_body, "freejoint")
    
    # Add geometries for base group (group 0)
    for idx in jsondata['group_info']['0']:
        part = parts_cfg[idx]
        ET.SubElement(base_body, "geom", attrib={
            "class": part["name"],
            "material": f'{part["name"]}_img'
        })
    
    # =========================================================================
    # Process kinematic groups (groups 1, 2, 3, ...)
    # =========================================================================
    have_free = 0  # Count of free joints
    dimscale = float(p.get("scale", "1 1 1").split(' ')[0])
    
    for group_idx in range(1, len(jsondata['group_info'])):
        joint_type = jsondata['group_info'][str(group_idx)][-1]  # Last element is joint type
        group_params = jsondata['group_info'][str(group_idx)][2]  # Joint parameters
        group_members = jsondata['group_info'][str(group_idx)][0]  # Part indices in this group
        
        # ---------------------------------------------------------------------
        # Type A: Free joint (floating, 6-DOF)
        # ---------------------------------------------------------------------
        if joint_type == 'A':
            have_free += 1
            # Free joints are handled separately after all other joints
        
        # ---------------------------------------------------------------------
        # Type B: Slide/Prismatic joint (1-DOF translation)
        # ---------------------------------------------------------------------
        elif joint_type == 'B':
            movable_body = ET.SubElement(world, "body", attrib={
                "name": f"grouppart_{group_idx}",
                "pos": "0 0 0"
            })
            
            # Add prismatic joint
            ET.SubElement(movable_body, "joint", attrib={
                "type": "slide",
                "name": f"slide_{group_idx}",
                "axis": " ".join(map(str, group_params[:3])),      # Direction vector
                "range": " ".join(map(str, group_params[6:8])),    # Motion limits
                "damping": "0.001",
                "frictionloss": "0.0",
                "stiffness": "0"
            })
            
            # Add part geometries
            for idx in group_members:
                part = parts_cfg[idx]
                ET.SubElement(movable_body, "geom", attrib={
                    "class": part["name"],
                    "material": f'{part["name"]}_img'
                })
        
        # ---------------------------------------------------------------------
        # Type C: Revolute/Hinge joint (1-DOF rotation)
        # ---------------------------------------------------------------------
        elif joint_type == 'C':
            movable_body = ET.SubElement(world, "body", attrib={
                "name": f"grouppart_{group_idx}",
                "pos": "0 0 0"
            })
            
            # Check if continuous (unlimited rotation) vs limited
            is_continuous = (group_params[6] == -1 and group_params[7] == 1)
            
            if is_continuous:
                # Continuous rotation (no limits)
                ET.SubElement(movable_body, "joint", attrib={
                    "type": "hinge",
                    "name": f"pivot_{group_idx}",
                    "axis": " ".join(map(str, group_params[:3])),
                    "pos": " ".join(map(str, (np.array(group_params[3:6]) * dimscale).tolist())),
                    "range": " ".join(map(str, (np.array([-3000, 3000]) * np.pi).tolist())),
                    "damping": "0.001",
                    "frictionloss": "0.0",
                    "stiffness": "0"
                })
            else:
                # Limited rotation
                ET.SubElement(movable_body, "joint", attrib={
                    "type": "hinge",
                    "name": f"pivot_{group_idx}",
                    "axis": " ".join(map(str, group_params[:3])),
                    "pos": " ".join(map(str, (np.array(group_params[3:6]) * dimscale).tolist())),
                    "range": " ".join(map(str, (np.array(group_params[6:8]) * np.pi).tolist())),
                    "damping": "0.001",
                    "frictionloss": "0.0",
                    "stiffness": "0"
                })
            
            # Add part geometries
            for idx in group_members:
                part = parts_cfg[idx]
                ET.SubElement(movable_body, "geom", attrib={
                    "class": part["name"],
                    "material": f'{part["name"]}_img'
                })
        
        # ---------------------------------------------------------------------
        # Type D: Ball joint (3-DOF rotation, spherical)
        # ---------------------------------------------------------------------
        elif joint_type == 'D':
            movable_body = ET.SubElement(world, "body", attrib={
                "name": f"grouppart_{group_idx}",
                "pos": "0 0 0"
            })
            
            ET.SubElement(movable_body, "joint", attrib={
                "type": "ball",
                "name": f"ball_{group_idx}",
                "pos": " ".join(map(str, (np.array(group_params[3:6]) * dimscale).tolist())),
                "damping": "0.001",
                "frictionloss": "0.0",
                "stiffness": "0"
            })
            
            # Add part geometries
            for idx in group_members:
                part = parts_cfg[idx]
                ET.SubElement(movable_body, "geom", attrib={
                    "class": part["name"],
                    "material": f'{part["name"]}_img'
                })
        
        # ---------------------------------------------------------------------
        # Type CB: Combined revolute + slide joint
        # ---------------------------------------------------------------------
        elif joint_type == 'CB':
            movable_body = ET.SubElement(world, "body", attrib={
                "name": f"grouppart_{group_idx}",
                "pos": "0 0 0"
            })
            
            # Add revolute joint first
            is_continuous = (group_params[6] == -1 and group_params[7] == 1)
            
            if is_continuous:
                ET.SubElement(movable_body, "joint", attrib={
                    "type": "hinge",
                    "name": f"pivot_{group_idx}",
                    "axis": " ".join(map(str, group_params[:3])),
                    "pos": " ".join(map(str, (np.array(group_params[3:6]) * dimscale).tolist())),
                    "range": " ".join(map(str, (np.array([-3000, 3000]) * np.pi).tolist())),
                    "damping": "0.001",
                    "frictionloss": "0.0",
                    "stiffness": "0"
                })
            else:
                ET.SubElement(movable_body, "joint", attrib={
                    "type": "hinge",
                    "name": f"pivot_{group_idx}",
                    "axis": " ".join(map(str, group_params[:3])),
                    "pos": " ".join(map(str, (np.array(group_params[3:6]) * dimscale).tolist())),
                    "range": " ".join(map(str, (np.array(group_params[6:8]) * np.pi).tolist())),
                    "damping": "0.001",
                    "frictionloss": "0.0",
                    "stiffness": "0"
                })
            
            # Add slide joint
            ET.SubElement(movable_body, "joint", attrib={
                "type": "slide",
                "name": f"slide_{group_idx}",
                "axis": " ".join(map(str, group_params[8:11])),   # Slide direction
                "range": " ".join(map(str, group_params[14:])),   # Slide limits
                "damping": "0.001",
                "frictionloss": "0.0",
                "stiffness": "0"
            })
            
            # Add part geometries
            for idx in group_members:
                part = parts_cfg[idx]
                ET.SubElement(movable_body, "geom", attrib={
                    "class": part["name"],
                    "material": f'{part["name"]}_img'
                })
    
    # =========================================================================
    # Handle free joints (Type A) separately - they need special placement
    # =========================================================================
    if have_free > 0:
        for group_idx in range(1, len(jsondata['group_info'])):
            if jsondata['group_info'][str(group_idx)][-1] == 'A':
                movable_body = ET.SubElement(world, "body", attrib={
                    "name": f"grouppart_{group_idx}",
                    "pos": "0 0 1",
                    "euler": base_euler
                })
                ET.SubElement(movable_body, "freejoint")
                
                for idx in jsondata['group_info'][str(group_idx)][0]:
                    part = parts_cfg[idx]
                    ET.SubElement(movable_body, "geom", attrib={
                        "class": part["name"],
                        "material": f'{part["name"]}_img'
                    })

    # =========================================================================
    # Reparent bodies according to kinematic hierarchy
    # =========================================================================
    reparent_by_group_info(
        mujoco, 
        jsondata['group_info'],
        base_body_name="base",
        group_body_prefix="grouppart_"
    )
    
    # =========================================================================
    # Handle deformable objects (if enabled)
    # =========================================================================
    if have_free > 0:
        for group_idx in range(1, len(jsondata['group_info'])):
            joint_type = jsondata['group_info'][str(group_idx)][-1]
            
            if joint_type == 'A' and deformable == 0:
                # Extract free body to world level (rigid)
                extract_body_to_world(mujoco, f"grouppart_{group_idx}")
            
            elif joint_type == 'A' and deformable == 1:
                # Create deformable (flex) object instead of rigid body
                world = find_worldbody(mujoco)
                target = find_body(mujoco, f"grouppart_{group_idx}")
                parent = find_parent(mujoco, target)
                parent.remove(target)

                # Get mesh info for flex computation
                filename = target.findall('geom')[0].get('class')
                meshid = filename.split('l_')[1].split('_')[0]

                # Calculate scaling from object dimensions
                str_list = jsondata['dimension'].split(' ')[0].split('*')
                sorted_list = sorted(str_list, key=float, reverse=True)
                scaling = float(sorted_list[0]) / 100

                # Compute mass from volume and density
                mesh = trimesh.load(os.path.join(
                    out_path.split('basic.xml')[0],
                    "./objs", str(meshid), f"{meshid}.obj"
                ))
                voxel_size = 0.01
                voxel_grid = mesh.voxelized(pitch=voxel_size)
                occupied = voxel_grid.matrix
                volume = np.sum(occupied) * (voxel_size ** 3)
                mass = volume * (scaling ** 3) * jsondata['parts'][int(meshid)]['density']

                # Create flexcomp element for deformable simulation
                flex = ET.SubElement(world, "flexcomp", attrib={
                    "type": "mesh",
                    "file": os.path.join("./objs", str(meshid), f"{meshid}.obj"),
                    "pos": "0 0 1",
                    "scale": p.get("scale", "1 1 1"),
                    "dim": "2",
                    "euler": "0 0 0",
                    "radius": "0.001",
                    "name": filename,
                    "dof": "trilinear",
                    "mass": str(mass),
                })
                
                # Add elasticity properties
                ET.SubElement(flex, "elasticity", attrib={
                    "young": str(float(jsondata['parts'][int(meshid)]["Young's Modulus (GPa)"]) * 1e9),
                    "poisson": str(jsondata['parts'][int(meshid)]["Poisson's Ratio"]),
                    "damping": "0.001"
                })
                
                # Add contact properties
                ET.SubElement(flex, "contact", attrib={
                    "selfcollide": "none",
                    "internal": "false",
                })

    # =========================================================================
    # Write MJCF file
    # =========================================================================
    _indent(mujoco)
    tree = ET.ElementTree(mujoco)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    
    return out_path


# =============================================================================
# MJCF BODY MANIPULATION UTILITIES
# =============================================================================

def find_worldbody(root: ET.Element) -> ET.Element:
    """Find the <worldbody> element in an MJCF tree."""
    for e in root.iter("worldbody"):
        return e
    raise ValueError("Cannot find <worldbody>")


def find_body(root: ET.Element, name: str) -> ET.Element | None:
    """Find a body by name in an MJCF tree."""
    for b in root.iter("body"):
        if b.get("name") == name:
            return b
    return None


def find_parent(root: ET.Element, node: ET.Element) -> ET.Element | None:
    """Find the parent element of a node in an XML tree."""
    for e in root.iter():
        for c in list(e):
            if c is node:
                return e
    return None


def is_direct_child_of_world(root: ET.Element, node: ET.Element) -> bool:
    """Check if a node is a direct child of worldbody."""
    parent = find_parent(root, node)
    return parent is not None and parent.tag == "worldbody"


def extract_body_to_world(root: ET.Element, body_name: str) -> bool:
    """
    Move a body from its current parent to directly under worldbody.
    
    Used for free-floating bodies that shouldn't be nested.
    
    Args:
        root (ET.Element): Root MJCF element
        body_name (str): Name of body to extract
    
    Returns:
        bool: True if moved, False if already at world level
    """
    world = find_worldbody(root)
    target = find_body(root, body_name)

    if target is None:
        raise ValueError(f"Cannot find body: {body_name}")

    if is_direct_child_of_world(root, target):
        print(f"'{body_name}' already at world level, skipping")
        return False

    parent = find_parent(root, target)
    if parent is None:
        raise RuntimeError("Cannot find the parent node")

    parent.remove(target)
    world.append(target)
    print(f"Moved '{body_name}' from '{parent.get('name')}' to <worldbody>")
    return True


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    """
    Main entry point for URDF/MJCF generation pipeline.
    
    This script processes 3D segmented meshes and VLM-annotated basic_info.txt
    to generate physics simulation-ready files:
    1. URDF (Universal Robot Description Format) for ROS/PyBullet
    2. MJCF (MuJoCo XML Format) for MuJoCo physics simulation
    
    The pipeline:
    1. Parse basic_info.txt to extract object metadata and group_info
    2. Optionally post-process joint positions using candidate generation
    3. Generate URDF with articulated joints
    4. Generate MJCF with physics properties
    
    Usage:
        python 4_simready_gen.py --basepath ./test_demo --fixed_base 1
    """
    
    # =========================================================================
    # Parse command line arguments
    # =========================================================================
    parser = argparse.ArgumentParser(
        description="Generate URDF and MJCF from segmented 3D models"
    )
    parser.add_argument(
        '--voxel_define', type=int, default=32,
        help='Voxel grid resolution (default: 32)'
    )
    parser.add_argument(
        '--basepath', type=str, default='./test_demo',
        help='Base path containing object folders'
    )
    parser.add_argument(
        '--process', type=int, default=0,
        help='Enable joint position post-processing (0=off, 1=on)'
    )
    parser.add_argument(
        '--fixed_base', type=int, default=0,
        help='Fix base in MJCF (0=floating, 1=fixed)'
    )
    parser.add_argument(
        '--deformable', type=int, default=0,
        help='Enable deformable objects in MJCF (0=rigid, 1=flex)'
    )
    args = parser.parse_args()
    
    # Initialize logging
    logger = get_logger(os.path.join('exp_urdf.log'), verbosity=1)
    logger.info('Starting URDF/MJCF generation pipeline')

    voxel_define = args.voxel_define
    basepath = args.basepath
    namelist = os.listdir(basepath)

    # =========================================================================
    # Process each object folder
    # =========================================================================
    for filename in namelist:
        logger.info(f'Processing: {filename}')
        
        # Skip folders without generated meshes
        if not os.path.exists(os.path.join(basepath, filename, 'objs')):
            logger.info(f'Skipping (no objs folder): {filename}')
            continue

        # =====================================================================
        # STEP 1: Parse basic_info.txt (VLM output)
        # =====================================================================
        with open(os.path.join(basepath, filename, 'basic_info.txt'), "r", encoding="utf-8") as f:
            basicqu = f.read()

        lines = [line.strip() for line in basicqu.strip().split('\n') if line.strip()]

        # Initialize data dictionary
        data = {}

        # Extract object metadata
        data['object_name'] = re.search(r'Name:\s*(.*)', lines[0]).group(1)
        data['category'] = re.search(r'Category:\s*(.*)', lines[1]).group(1)
        data['dimension'] = re.search(r'Dimension:\s*(.*)', lines[2]).group(1)

        # -----------------------------------------------------------------
        # Parse part definitions (l_0, l_1, l_2, ...)
        # Format: l_N: name, priority, material, density, young, poisson, desc
        # -----------------------------------------------------------------
        parts = []
        for line in lines:
            if line.startswith("l_"):
                match = re.match(
                    r'l_(\d+):\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*(.*)',
                    line
                )
                if match:
                    parts.append({
                        "label": int(match.group(1)),
                        "name": match.group(2).strip(),
                        "priority_rank": int(match.group(3)),
                        "material": match.group(4).strip(),
                        "density": match.group(5).strip(),
                        "Young's Modulus (GPa)": match.group(6).strip(),
                        "Poisson's Ratio": match.group(7).strip(),
                        "Basic_description": match.group(8).strip()
                    })

        data['parts'] = parts

        # -----------------------------------------------------------------
        # Parse group definitions (kinematic groups with joint info)
        # Format: group_N: [members], relative to group_M, Type: X, params...
        # -----------------------------------------------------------------
        group_info = {}
        
        for i, line in enumerate(lines):
            if not re.match(r'^group_\d+\s*:', line.strip(), flags=re.IGNORECASE):
                continue

            # Extract group ID and members
            gm = re.search(r'group_(\d+):\s*\[(.*?)\]', line, flags=re.IGNORECASE)
            if not gm:
                continue
            
            gid = gm.group(1)
            members_raw = gm.group(2)

            # Parse member part IDs (e.g., "l_0, l_1" -> [0, 1])
            members = []
            for tok in members_raw.split(','):
                tok = tok.strip().strip("'").strip('"')
                nm = re.search(r'l_(\d+)', tok, flags=re.IGNORECASE)
                if nm:
                    members.append(int(nm.group(1)))

            # Extract joint type (A, B, C, D, CB, or E=fixed)
            tm = re.search(r'Type:\s*([A-Za-z])', line, flags=re.IGNORECASE)
            gtype = tm.group(1).upper() if tm else "E"
            if ': CB' in line:
                gtype = 'CB'

            # Extract relative-to parent group
            rel_idx = None
            rel_matches = re.findall(
                r'(?:relative\s*to\s*)+group_(\d+)', line, flags=re.IGNORECASE
            )
            if rel_matches:
                rel_idx = int(rel_matches[-1])

            # Initialize parameter vector (direction, position, range)
            param_vec = [0.0] * 8

            # ---------------------------------------------------------
            # Parse joint parameters based on type
            # ---------------------------------------------------------
            if gtype not in ("E", "A", "CB"):
                # Types B, C, D: direction[3], position[3], range[2]
                scan_block = line

                dir_v = _extract_bracket_list(scan_block, 'direction', 3)
                pos_v = _extract_bracket_list(scan_block, 'position', 3)

                # Normalize position to [-0.5, 0.5] range
                pos_v = ((np.array(pos_v)) / voxel_define - 0.5).tolist()
                
                # Parse range based on joint type
                if gtype == "C":
                    # Revolute: convert degrees to pi-normalized
                    rng_v = _extract_bracket_list(scan_block, 'range', 2)
                    rng_v = (np.array(rng_v) / 180).tolist()

                if gtype == "B":
                    # Prismatic: normalize to voxel scale
                    rng_v = _extract_bracket_list(scan_block, 'range', 2)
                    rng_v = (np.array(rng_v) / voxel_define).tolist()

                param_vec = dir_v + pos_v + rng_v

            elif gtype == "CB":
                # Combined type: revolute + slide parameters
                scan_block = line

                # Revolute axis parameters
                dir_v = _extract_bracket_list(scan_block, 'axis direction', 3)
                pos_v = _extract_bracket_list(scan_block, 'axis position', 3)
                pos_v = ((np.array(pos_v)) / voxel_define - 0.5).tolist()
                
                rng_v = _extract_bracket_list(scan_block, 'revolute range', 2)
                rng_v = (np.array(rng_v) / 180).tolist()

                # Slide parameters
                dir_v1 = _extract_bracket_list(scan_block, 'slide direction', 3)
                rng_v1 = _extract_bracket_list(scan_block, 'slide range', 2)
                rng_v1 = (np.array(rng_v1) / voxel_define).tolist()

                # Combined: [rev_dir, rev_pos, rev_range, slide_dir, pad, slide_range]
                param_vec = dir_v + pos_v + rng_v + dir_v1 + [0, 0, 0] + rng_v1

            # Store group info
            if gid == str(0):
                # Group 0 is base (fixed parts), just store members
                group_info[gid] = members
            else:
                # Movable groups: [members, parent_idx, params, type]
                group_info[gid] = [members, str(rel_idx), param_vec, gtype]

        data['group_info'] = group_info

        # =====================================================================
        # STEP 2: Post-process joint positions (optional)
        # =====================================================================
        # When --process=1, refine joint positions using geometry-based
        # candidate detection to snap to actual mesh boundaries
        
        if args.process:
            for group_id in range(1, len(group_info)):
                joint_type = group_info[str(group_id)][-1]
                
                # ---------------------------------------------------------
                # Refine revolute (C) and combined (CB) joint positions
                # ---------------------------------------------------------
                if joint_type in ('C', 'CB'):
                    # Get parent group's members
                    parent_id = group_info[str(group_id)][1]
                    if parent_id == '0':
                        group_b = group_info['0']
                    else:
                        parent_data = group_info[parent_id]
                        group_b = parent_data if isinstance(parent_data, list) and isinstance(parent_data[0], int) else parent_data[0]

                    # Generate candidate axis positions from mesh intersection
                    allcandidate = generate_allcandidate(
                        group_info[str(group_id)][0],  # Current group members
                        group_b,                        # Parent group members  
                        os.path.join(basepath, filename)
                    )

                    # Compute weights: ignore axis direction dimension
                    axisdir = np.array(group_info[str(group_id)][2][:3])
                    axisdir = np.int32(axisdir / np.linalg.norm(axisdir))
                    weights = np.array([1, 1, 1])
                    weights[np.where(axisdir == 1)] = 0
                    
                    # Find nearest candidate point
                    current_pos = np.array(group_info[str(group_id)][2][3:6])
                    error = (allcandidate - current_pos) * weights
                    dist = np.linalg.norm(error, axis=1)
                    idx = np.argmin(dist)
                    nearest_point = allcandidate[idx]

                    # Snap if close enough (threshold: 0.03)
                    if np.linalg.norm((nearest_point - current_pos) * weights) < 0.03:
                        group_info[str(group_id)][2][3:6] = nearest_point.tolist()

                # ---------------------------------------------------------
                # Refine ball joint (D) center positions
                # ---------------------------------------------------------
                if joint_type == 'D':
                    # Get parent group's members
                    parent_id = group_info[str(group_id)][1]
                    if parent_id == '0':
                        group_b = group_info['0']
                    else:
                        parent_data = group_info[parent_id]
                        group_b = parent_data if isinstance(parent_data, list) and isinstance(parent_data[0], int) else parent_data[0]

                    # Generate candidate center positions
                    allcandidate = generate_allcandidate_center(
                        group_info[str(group_id)][0],
                        group_b,
                        os.path.join(basepath, filename)
                    )

                    # Find nearest candidate
                    weights = np.array([1, 1, 1])
                    current_pos = np.array(group_info[str(group_id)][2][3:6])
                    error = (allcandidate - current_pos) * weights
                    dist = np.linalg.norm(error)
                    idx = np.argmin(dist)
                    nearest_point = allcandidate[idx]

                    # Snap if close enough
                    if np.linalg.norm((nearest_point - current_pos) * weights) < 0.03:
                        group_info[str(group_id)][2][3:6] = nearest_point.tolist()

        # =====================================================================
        # STEP 3: Save processed data as JSON
        # =====================================================================
        with open(os.path.join(basepath, filename, 'basic_info.json'), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        # =====================================================================
        # STEP 4: Generate URDF file
        # =====================================================================
        jsonfile = os.path.join(basepath, filename, 'basic_info.json')
        geofile = os.path.join(basepath, filename, 'objs')

        with open(jsonfile, 'r') as fp:
            jsondata = json.load(fp)

        mov = jsondata['group_info']

        # Create root URDF element
        robot = ET.Element('robot', name='scene')
        
        # World link (fixed reference frame)
        link = ET.SubElement(robot, 'link', name='l_world')
        add_inertial(link)

        save = 1  # Track number of movable joints

        # -----------------------------------------------------------------
        # Case 1: Static object (only base group, no movable parts)
        # -----------------------------------------------------------------
        if len(mov) == 1:
            fixlist = mov['0']
            
            # Create links for each part in base group
            for fixindex in fixlist:
                link = ET.SubElement(robot, 'link', name=f'l_{fixindex}')
                add_inertial(link)
                
                mesh_path = os.path.join(geofile, str(fixindex), f'{fixindex}.obj')
                if os.path.exists(mesh_path):
                    visual = ET.SubElement(link, 'visual')
                    geometry = ET.SubElement(visual, "geometry")
                    ET.SubElement(
                        geometry, "mesh",
                        filename=os.path.join('./objs', str(fixindex), f'{fixindex}.obj'),
                        scale="1 1 1"
                    )
                    ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")

            # Chain parts with fixed joints
            for i in range(len(fixlist) - 1):
                parentname = f'l_{fixlist[i]}'
                childname = f'l_{fixlist[i+1]}'
                add_fixed_joint(
                    robot, f'joint_fixed_{fixlist[i]}_{fixlist[i+1]}',
                    parentname, childname, xyz="0 0 0", rpy="0 0 0"
                )
            
            # Connect first part to world
            add_fixed_joint(
                robot, f'joint_fixed_world{fixlist[0]}',
                'l_world', f'l_{fixlist[0]}', xyz="0 0 0", rpy="0 0 0"
            )

        # -----------------------------------------------------------------
        # Case 2: Articulated object (multiple kinematic groups)
        # -----------------------------------------------------------------
        else:
            offset = False

            # Create base group links
            fixlist = mov['0']
            for fixindex in fixlist:
                link = ET.SubElement(robot, 'link', name=f'l_{fixindex}')
                add_inertial(link)
                
                mesh_path = os.path.join(geofile, str(fixindex), f'{fixindex}.obj')
                if os.path.exists(mesh_path):
                    visual = ET.SubElement(link, 'visual')
                    geometry = ET.SubElement(visual, "geometry")
                    ET.SubElement(
                        geometry, "mesh",
                        filename=os.path.join('./objs', str(fixindex), f'{fixindex}.obj'),
                        scale="1 1 1"
                    )
                    ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")

            # Chain base parts
            for i in range(len(fixlist) - 1):
                parentname = f'l_{fixlist[i]}'
                childname = f'l_{fixlist[i+1]}'
                add_fixed_joint(
                    robot, f'joint_fixed_{fixlist[i]}_{fixlist[i+1]}',
                    parentname, childname, xyz="0 0 0", rpy="0 0 0"
                )
            
            add_fixed_joint(
                robot, f'joint_fixed_world{fixlist[0]}',
                'l_world', f'l_{fixlist[0]}', xyz="0 0 0", rpy="0 0 0"
            )

            # -------------------------------------------------------------
            # Process movable groups (groups 1, 2, 3, ...)
            # -------------------------------------------------------------
            groupnum = len(mov)
            for groupindex in range(1, groupnum):
                fixlist = mov[str(groupindex)][0]
                
                # Create links for parts in this group
                for fixindex in fixlist:
                    link = ET.SubElement(robot, 'link', name=f'l_{fixindex}')
                    add_inertial(link)
                    
                    mesh_path = os.path.join(geofile, str(fixindex), f'{fixindex}.obj')
                    if os.path.exists(mesh_path):
                        visual = ET.SubElement(link, 'visual')
                        geometry = ET.SubElement(visual, "geometry")
                        ET.SubElement(
                            geometry, "mesh",
                            filename=os.path.join('./objs', str(fixindex), f'{fixindex}.obj'),
                            scale="1 1 1"
                        )
                        ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")

                # Chain parts within this group with fixed joints
                for i in range(len(fixlist) - 1):
                    parentname = f'l_{fixlist[i]}'
                    childname = f'l_{fixlist[i+1]}'
                    add_fixed_joint(
                        robot, f'joint_fixed_{fixlist[i]}_{fixlist[i+1]}',
                        parentname, childname, xyz="0 0 0", rpy="0 0 0"
                    )
                
                # Determine parent and child for this kinematic group
                parent_group_data = mov[mov[str(groupindex)][1]]
                if isinstance(parent_group_data[0], int):
                    parentgroupindex = str(parent_group_data[0])
                else:
                    parentgroupindex = str(parent_group_data[0][0])

                childgroupindex = fixlist[0]
                parentgroupname = f'l_{parentgroupindex}'
                childgroupname = f'l_{childgroupindex}'

                # Create abstract link for joint connection
                abs_link = ET.SubElement(
                    robot, 'link',
                    name=f'abstract_{parentgroupindex}_{childgroupindex}'
                )
                add_inertial(abs_link)

                joint_type = mov[str(groupindex)][-1]
                params = mov[str(groupindex)][-2]

                # ---------------------------------------------------------
                # JOINT TYPE A: Free/Floating (6-DOF)
                # ---------------------------------------------------------
                if joint_type == 'A':
                    add_fixed_joint(
                        robot,
                        f'joint_fixed_abstract_{parentgroupindex}_{childgroupindex}',
                        f'abstract_{parentgroupindex}_{childgroupindex}',
                        childgroupname, xyz="0 0 0", rpy="0 0 0"
                    )

                    joint = ET.SubElement(
                        robot, "joint",
                        name=f'joint_free_{parentgroupname}_abstract_{parentgroupindex}_{childgroupindex}',
                        type="floating"
                    )
                    ET.SubElement(joint, "parent", link=parentgroupname)
                    ET.SubElement(joint, "child", link=f'abstract_{parentgroupindex}_{childgroupindex}')
                    ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")

                # ---------------------------------------------------------
                # JOINT TYPE B: Prismatic/Slide (1-DOF translation)
                # ---------------------------------------------------------
                elif joint_type == 'B':
                    save += 1
                    add_fixed_joint(
                        robot,
                        f'joint_fixed_abstract_{parentgroupindex}_{childgroupindex}',
                        f'abstract_{parentgroupindex}_{childgroupindex}',
                        childgroupname, xyz="0 0 0", rpy="0 0 0"
                    )

                    # Slide axis direction
                    xyz = f'{params[0]} {params[1]} {params[2]}'

                    joint = ET.SubElement(
                        robot, "joint",
                        name=f'joint_prismatic_{parentgroupname}_abstract_{parentgroupindex}_{childgroupindex}',
                        type="prismatic"
                    )
                    ET.SubElement(joint, "parent", link=parentgroupname)
                    ET.SubElement(joint, "child", link=f'abstract_{parentgroupindex}_{childgroupindex}')
                    ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
                    ET.SubElement(joint, "axis", xyz=xyz)
                    ET.SubElement(
                        joint, "limit",
                        lower=str(params[-2]), upper=str(params[-1]),
                        effort="2000.0", velocity="2.0"
                    )

                # ---------------------------------------------------------
                # JOINT TYPE C: Revolute/Hinge (1-DOF rotation)
                # ---------------------------------------------------------
                elif joint_type == 'C':
                    save += 1
                    
                    # Axis position and negative for child offset
                    point = f'{params[3]} {params[4]} {params[5]}'
                    pointrev = f'{-params[3]} {-params[4]} {-params[5]}'
                    xyz = f'{params[0]} {params[1]} {params[2]}'

                    add_fixed_joint(
                        robot,
                        f'joint_fixed_abstract_{parentgroupindex}_{childgroupindex}',
                        f'abstract_{parentgroupindex}_{childgroupindex}',
                        childgroupname, xyz=pointrev, rpy="0 0 0"
                    )

                    # Check if continuous (unlimited range)
                    is_continuous = (params[-2] == -1 and params[-1] == 1)
                    
                    joint = ET.SubElement(
                        robot, "joint",
                        name=f'joint_revolute_{parentgroupname}_abstract_{parentgroupindex}_{childgroupindex}',
                        type="continuous" if is_continuous else "revolute"
                    )
                    ET.SubElement(joint, "parent", link=parentgroupname)
                    ET.SubElement(joint, "child", link=f'abstract_{parentgroupindex}_{childgroupindex}')
                    ET.SubElement(joint, "origin", xyz=point, rpy="0 0 0")
                    ET.SubElement(joint, "axis", xyz=xyz)

                    if is_continuous:
                        ET.SubElement(joint, "limit", effort="2000.0", velocity="2.0")
                    else:
                        ET.SubElement(
                            joint, "limit",
                            lower=str(params[-2] * np.pi),
                            upper=str(params[-1] * np.pi),
                            effort="2000.0", velocity="2.0"
                        )

                # ---------------------------------------------------------
                # JOINT TYPE D: Ball/Spherical (3-DOF rotation)
                # Implemented as 3 chained revolute joints (ZXY Euler)
                # ---------------------------------------------------------
                elif joint_type == 'D':
                    save += 1

                    point = f'{params[3]} {params[4]} {params[5]}'
                    pointrev = f'{-params[3]} {-params[4]} {-params[5]}'
                    xyz = f'{params[0]} {params[1]} {params[2]}'

                    add_fixed_joint(
                        robot,
                        f'joint_fixed_abstract_{parentgroupindex}_{childgroupindex}',
                        f'abstract_{parentgroupindex}_{childgroupindex}',
                        childgroupname, xyz=pointrev, rpy="0 0 0"
                    )

                    # Create intermediate links for ball joint decomposition
                    abs_linkx = ET.SubElement(
                        robot, 'link',
                        name=f'abstract_x_{parentgroupindex}_{childgroupindex}'
                    )
                    add_inertial(abs_linkx, pointrev)
                    
                    abs_linkz = ET.SubElement(
                        robot, 'link',
                        name=f'abstract_z_{parentgroupindex}_{childgroupindex}'
                    )
                    add_inertial(abs_linkz, pointrev)

                    # First rotation joint (Z-axis)
                    joint = ET.SubElement(
                        robot, "joint",
                        name=f'joint_hinge_y_{parentgroupname}_abstract_{parentgroupindex}_{childgroupindex}',
                        type="revolute"
                    )
                    ET.SubElement(joint, "parent", link=parentgroupname)
                    ET.SubElement(joint, "child", link=f'abstract_z_{parentgroupindex}_{childgroupindex}')
                    ET.SubElement(joint, "origin", xyz=point, rpy="0 0 0")
                    ET.SubElement(joint, "axis", xyz="0 0 1")
                    ET.SubElement(
                        joint, "limit",
                        lower=str(-np.pi), upper=str(np.pi),
                        effort="2000.0", velocity="2.0"
                    )

                    # Second rotation joint (X-axis)
                    joint = ET.SubElement(
                        robot, "joint",
                        name=f'joint_hinge_z_{parentgroupname}_abstract_{parentgroupindex}_{childgroupindex}',
                        type="revolute"
                    )
                    ET.SubElement(joint, "parent", link=f'abstract_z_{parentgroupindex}_{childgroupindex}')
                    ET.SubElement(joint, "child", link=f'abstract_x_{parentgroupindex}_{childgroupindex}')
                    ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
                    ET.SubElement(joint, "axis", xyz="1 0 0")
                    ET.SubElement(
                        joint, "limit",
                        lower=str(-np.pi), upper=str(np.pi),
                        effort="2000.0", velocity="2.0"
                    )

                    # Third rotation joint (Y-axis)
                    joint = ET.SubElement(
                        robot, "joint",
                        name=f'joint_hinge_x_{parentgroupname}_abstract_{parentgroupindex}_{childgroupindex}',
                        type="revolute"
                    )
                    ET.SubElement(joint, "parent", link=f'abstract_x_{parentgroupindex}_{childgroupindex}')
                    ET.SubElement(joint, "child", link=f'abstract_{parentgroupindex}_{childgroupindex}')
                    ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
                    ET.SubElement(joint, "axis", xyz="0 1 0")
                    ET.SubElement(
                        joint, "limit",
                        lower=str(-np.pi), upper=str(np.pi),
                        effort="2000.0", velocity="2.0"
                    )

                # ---------------------------------------------------------
                # JOINT TYPE CB: Combined Revolute + Prismatic
                # ---------------------------------------------------------
                elif joint_type == 'CB':
                    save += 1

                    # Axis position
                    point = f'{params[3]} {params[4]} {params[5]}'
                    pointrev = f'{-params[3]} {-params[4]} {-params[5]}'
                    
                    # Revolute axis direction
                    xyz = f'{params[0]} {params[1]} {params[2]}'
                    
                    # Slide axis direction (params[8:11])
                    xyz1 = f'{params[8]} {params[9]} {params[10]}'

                    add_fixed_joint(
                        robot,
                        f'joint_fixed_abstract_{parentgroupindex}_{childgroupindex}',
                        f'abstract_{parentgroupindex}_{childgroupindex}',
                        childgroupname, xyz=pointrev, rpy="0 0 0"
                    )

                    # Intermediate link for combined joint
                    abs_linkx = ET.SubElement(
                        robot, 'link',
                        name=f'abstract_x_{parentgroupindex}_{childgroupindex}'
                    )
                    add_inertial(abs_linkx)

                    # Prismatic joint first
                    joint = ET.SubElement(
                        robot, "joint",
                        name=f'joint_prim_y_{parentgroupname}_abstract_{parentgroupindex}_{childgroupindex}',
                        type="prismatic"
                    )
                    ET.SubElement(joint, "parent", link=parentgroupname)
                    ET.SubElement(joint, "child", link=f'abstract_x_{parentgroupindex}_{childgroupindex}')
                    ET.SubElement(joint, "origin", xyz=point, rpy="0 0 0")
                    ET.SubElement(joint, "axis", xyz=xyz1)
                    ET.SubElement(
                        joint, "limit",
                        lower=str(params[-2]), upper=str(params[-1]),
                        effort="2000.0", velocity="2.0"
                    )

                    # Revolute joint second
                    is_continuous = (params[6] == -1 and params[7] == 1)
                    
                    joint = ET.SubElement(
                        robot, "joint",
                        name=f'joint_revo_x_{parentgroupname}_abstract_{parentgroupindex}_{childgroupindex}',
                        type="continuous" if is_continuous else "revolute"
                    )
                    ET.SubElement(joint, "parent", link=f'abstract_x_{parentgroupindex}_{childgroupindex}')
                    ET.SubElement(joint, "child", link=f'abstract_{parentgroupindex}_{childgroupindex}')
                    ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
                    ET.SubElement(joint, "axis", xyz=xyz)

                    if is_continuous:
                        ET.SubElement(joint, "limit", effort="2000.0", velocity="2.0")
                    else:
                        ET.SubElement(
                            joint, "limit",
                            lower=str(params[6] * np.pi),
                            upper=str(params[7] * np.pi),
                            effort="2000.0", velocity="2.0"
                        )

                # Unknown joint type
                else:
                    print(f'Error: Unknown joint type: {joint_type}')

        # =====================================================================
        # STEP 5: Write URDF file
        # =====================================================================
        tree = ET.ElementTree(robot)
        ET.indent(tree, space="  ", level=0)
        tree.write(
            os.path.join(basepath, filename, 'basic.urdf'),
            encoding="utf-8", xml_declaration=True
        )

        # =====================================================================
        # STEP 6: Generate MJCF file
        # =====================================================================
        parts_cfg = jsondata['parts']

        # Calculate model scale from dimensions
        nums = [int(x) for x in re.findall(r'\d+', jsondata['dimension'])]
        max_num = max(nums) / 100  # Convert to meters

        # Prepare part configurations for MJCF
        for partind in range(len(parts_cfg)):
            part = parts_cfg[partind]
            part['name'] = f'l_{part["label"]}_{part["name"]}'
            part['mesh_file'] = os.path.join('./objs', str(partind), f'{partind}.obj')
            part['scale'] = f'{max_num} {max_num} {max_num}'
            part['tex_file'] = os.path.join('./objs', str(partind), 'material_0.png')
            
            # Convert density from g/cm³ to kg/m³
            part['density'] = float(part['density'].split('g/cm')[0]) * 1000
            part['fluidshape'] = 'ellipsoid'
            part['contype'] = '1'
            part['conaffinity'] = '1'

        # Copy skybox texture
        shutil.copy(
            'mjcf_source/desert.png',
            os.path.join(basepath, filename, 'desert.png')
        )

        # Generate MJCF
        out = generate_mjcf(
            jsondata=jsondata,
            out_path=os.path.join(basepath, filename, 'basic.xml'),
            parts=parts_cfg,
            fixed_base=args.fixed_base,
            deformable=args.deformable
        )

        logger.info(f'Completed: {filename}')




    