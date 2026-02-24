"""
===============================================================================
3_split.py - Mesh Segmentation Using Geodesic Distance Propagation
===============================================================================

This script segments a 3D mesh into multiple parts based on voxel labels from
the VLM output. Each part is exported as a separate OBJ file.

Pipeline Overview:
    1. Load the generated GLB mesh and voxel part labels
    2. Find nearest labeled voxel for each mesh vertex
    3. Propagate labels using geodesic (surface) distance
    4. Assign face labels based on vertex majority voting
    5. Export each labeled region as a separate mesh

Algorithm Details:
    - Geodesic Propagation: Uses Dijkstra's algorithm on the mesh edge graph
    - Label Assignment: Vertices close to labeled voxels become "seed" vertices
    - Face Labeling: Each face takes the majority label of its vertices
    - Tie Breaking: Uses geodesic distance sum when votes are tied

Dependencies:
    - trimesh: Mesh loading and manipulation
    - scipy: KD-tree for nearest neighbor queries

Author: PhysX-Anything Team
===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import heapq
import logging
import argparse

import numpy as np
import trimesh
from scipy.spatial import cKDTree


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_edge_graph(mesh: trimesh.Trimesh):
    """
    Build an edge graph from mesh connectivity for geodesic distance computation.
    
    Creates adjacency lists where each vertex stores its neighbors and edge weights
    (Euclidean edge lengths). This graph is used for Dijkstra's algorithm.
    
    Args:
        mesh (trimesh.Trimesh): Input mesh
    
    Returns:
        tuple: (neighbors, weights) where:
            - neighbors: List of arrays, neighbors[v] = adjacent vertex indices
            - weights: List of arrays, weights[v] = edge lengths to neighbors[v]
    """
    edges = mesh.edges_unique
    V = mesh.vertices
    
    # Initialize adjacency lists for each vertex
    neighbors = [[] for _ in range(len(V))]
    weights = [[] for _ in range(len(V))]
    
    # Calculate edge lengths
    e_len = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
    
    # Build undirected graph (add both directions)
    for (u, v), w in zip(edges, e_len):
        neighbors[u].append(v)
        weights[u].append(w)
        neighbors[v].append(u)
        weights[v].append(w)
    
    # Convert to numpy arrays for faster access
    neighbors = [np.asarray(n, dtype=np.int64) for n in neighbors]
    weights = [np.asarray(w, dtype=np.float64) for w in weights]
    
    return neighbors, weights


# =============================================================================
# NEAREST LABEL COMPUTATION
# =============================================================================

def nearest_label_all_vertices(vertices, label_to_points):
    """
    Find the nearest labeled point for each mesh vertex using KD-trees.
    
    For each vertex, finds which label's point cloud contains the closest point.
    This is used to initialize seed labels for geodesic propagation.
    
    Args:
        vertices (np.ndarray): Mesh vertices, shape (N, 3)
        label_to_points (dict): Maps label (str) -> point coordinates (N, 3)
    
    Returns:
        tuple: (nearest_label, dmin_per_v, trees) where:
            - nearest_label: Label of nearest point for each vertex
            - dmin_per_v: Distance to nearest labeled point
            - trees: Dict of KD-trees for each label
    """
    # Build KD-tree for each label's point cloud
    trees = {}
    labels_sorted = sorted(label_to_points.keys(), key=lambda x: int(x))
    
    for lab in labels_sorted:
        P = np.asarray(label_to_points[lab], dtype=np.float64)
        trees[lab] = cKDTree(P) if len(P) > 0 else None

    # Initialize arrays
    V = vertices.shape[0]
    nearest_label = np.zeros(V, dtype=np.int64)
    dmin_per_v = np.full(V, np.inf, dtype=np.float64)

    # Find nearest label for each vertex
    for lab in labels_sorted:
        tree = trees[lab]
        if tree is None:
            continue
        
        # Query distance to nearest point in this label's cloud
        d, _ = tree.query(vertices, k=1, workers=-1)
        
        # Update if this label is closer
        mask = d < dmin_per_v
        dmin_per_v[mask] = d[mask]
        nearest_label[mask] = int(lab)

    return nearest_label, dmin_per_v, trees


# =============================================================================
# GEODESIC LABEL PROPAGATION
# =============================================================================

def multisource_geodesic_propagation_with_fallback(
    neighbors, weights, seed_mask, seed_labels, fallback_labels
):
    """
    Propagate labels from seed vertices along mesh surface using Dijkstra's algorithm.
    
    This is a multi-source shortest path algorithm where each seed vertex starts
    with distance 0. Labels are propagated to all vertices based on geodesic distance.
    
    Algorithm:
        1. Initialize seeds with distance 0 and their assigned labels
        2. Run Dijkstra's algorithm, propagating labels along shortest paths
        3. Use fallback labels for any unreachable vertices
    
    Args:
        neighbors: Adjacency list from build_edge_graph()
        weights: Edge weights from build_edge_graph()
        seed_mask (np.ndarray): Boolean mask indicating seed vertices
        seed_labels (np.ndarray): Labels for seed vertices
        fallback_labels (np.ndarray): Backup labels for unreachable vertices
    
    Returns:
        tuple: (labels, dist) where:
            - labels: Final label assignment for each vertex
            - dist: Geodesic distance to nearest seed of same label
    """
    V = len(neighbors)
    labels = np.full(V, -1, dtype=np.int64)
    dist = np.full(V, np.inf, dtype=np.float64)
    pq = []  # Priority queue: (distance, vertex)

    # Initialize seed vertices
    for v in range(V):
        if seed_mask[v]:
            labels[v] = seed_labels[v]
            dist[v] = 0.0
            heapq.heappush(pq, (0.0, v))

    # Handle case with no seeds
    if len(pq) == 0:
        return fallback_labels.copy(), np.zeros(V, dtype=np.float64)

    # Dijkstra's algorithm - propagate labels along shortest paths
    while pq:
        d_u, u = heapq.heappop(pq)
        
        # Skip if we've already found a shorter path to this vertex
        if d_u != dist[u]:
            continue
        
        lab_u = labels[u]
        
        # Relax edges to neighbors
        for nv, w in zip(neighbors[u], weights[u]):
            nd = d_u + w
            if nd < dist[nv]:
                dist[nv] = nd
                labels[nv] = lab_u
                heapq.heappush(pq, (nd, nv))

    # Apply fallback labels to any unlabeled vertices
    miss = (labels == -1)
    if np.any(miss):
        labels[miss] = fallback_labels[miss]
        dist[miss] = 0.0

    return labels, dist


# =============================================================================
# FACE LABELING
# =============================================================================

def face_majority_label(mesh: trimesh.Trimesh, vlabels, vdist):
    """
    Assign labels to faces based on majority voting of vertex labels.
    
    Each face's label is determined by:
        1. If all 3 vertices have the same label -> use that label
        2. Otherwise, use the most common label among vertices
        3. If tied, use the label with minimum total geodesic distance
    
    Args:
        mesh (trimesh.Trimesh): Input mesh
        vlabels (np.ndarray): Label for each vertex
        vdist (np.ndarray): Geodesic distance for each vertex
    
    Returns:
        np.ndarray: Label for each face
    """
    F = mesh.faces.shape[0]
    flabels = np.zeros(F, dtype=np.int64)
    
    for i in range(F):
        # Get the 3 vertices of this face
        vs = mesh.faces[i]
        labs = vlabels[vs]
        
        # Count votes for each label
        vals, counts = np.unique(labs, return_counts=True)
        
        if len(vals) == 1:
            # All vertices have same label - easy case
            flabels[i] = vals[0]
        else:
            # Multiple labels - use majority voting
            idx = np.argmax(counts)
            
            # Check if there's a clear winner
            if np.sum(counts == counts[idx]) == 1:
                flabels[i] = vals[idx]
            else:
                # Tie-breaker: use label with minimum total geodesic distance
                best_lab, best_sum = None, np.inf
                for lab in vals:
                    s = vdist[vs][labs == lab].sum()
                    if s < best_sum:
                        best_sum, best_lab = s, lab
                flabels[i] = best_lab
    
    return flabels


# =============================================================================
# ENSURING ALL LABELS HAVE FACES
# =============================================================================

def ensure_nonempty_per_label(mesh, flabels, label_to_points, min_faces=10):
    """
    Ensure every label has at least some faces assigned.
    
    If a label has no faces (due to poor voxel predictions), force-assign
    faces near the label's point cloud center.
    
    Args:
        mesh (trimesh.Trimesh): Input mesh
        flabels (np.ndarray): Current face labels (modified in-place)
        label_to_points (dict): Maps label -> point coordinates
        min_faces (int): Minimum faces to assign to missing labels
    
    Returns:
        np.ndarray: Updated face labels
    """
    labels_sorted = sorted(label_to_points.keys(), key=lambda x: int(x))
    F = mesh.faces.shape[0]
    
    # Build face adjacency for region growing
    adj = mesh.face_adjacency  # (M, 2) pairs of adjacent faces
    face_nbrs = [[] for _ in range(F)]
    for a, b in adj:
        face_nbrs[a].append(b)
        face_nbrs[b].append(a)

    tri_centers = mesh.triangles_center

    # Check each label
    for lab in labels_sorted:
        lab_i = int(lab)
        
        # Skip if this label already has faces
        if np.any(flabels == lab_i):
            continue

        P = np.asarray(label_to_points[lab], dtype=np.float64)
        if len(P) == 0:
            continue

        # Find face closest to the point cloud center
        c = P.mean(axis=0)
        idx0 = np.argmin(np.linalg.norm(tri_centers - c[None, :], axis=1))

        # Grow region from seed face using BFS
        picked = set([idx0])
        frontier = [idx0]
        
        while len(picked) < min_faces and frontier:
            new_frontier = []
            for f in frontier:
                for g in face_nbrs[f]:
                    if g not in picked:
                        picked.add(g)
                        new_frontier.append(g)
            frontier = new_frontier
        
        # Assign label to picked faces
        flabels[list(picked)] = lab_i

    return flabels


# =============================================================================
# MESH EXPORT
# =============================================================================

def export_label_submeshes(mesh: trimesh.Trimesh, flabels, out_dir):
    """
    Export each labeled region as a separate OBJ file.
    
    Args:
        mesh (trimesh.Trimesh): Input mesh
        flabels (np.ndarray): Label for each face
        out_dir (str): Output directory path
    """
    os.makedirs(out_dir, exist_ok=True)
    unique_labs = np.unique(flabels)
    
    for lab in unique_labs:
        mask = (flabels == lab)
        if not np.any(mask):
            continue
        
        # Extract submesh for this label
        sub = mesh.submesh([np.nonzero(mask)[0]], append=True, repair=True)
        
        if sub.vertices.shape[0] == 0 or sub.faces.shape[0] == 0:
            continue

        # Create subdirectory and export
        os.makedirs(os.path.join(out_dir, f"{lab}"), exist_ok=True)
        export_path = os.path.join(out_dir, f"{lab}", f"{lab}.obj")
        sub.export(export_path)
        
        print(f"[+] Saved: {export_path}  (V={len(sub.vertices)}, F={len(sub.faces)})")


# =============================================================================
# MAIN SEGMENTATION FUNCTION
# =============================================================================

def segment_mesh_by_wrapped_pcd_no_minus1(
    mesh,
    label_to_points: dict,
    out_dir: str = "out_submeshes",
    seed_tau_ratio: float = 0.02,
    min_seed_faces: int = 20
):
    """
    Segment a mesh into parts using voxel-based labels and geodesic propagation.
    
    This is the main entry point for mesh segmentation. It:
        1. Computes nearest voxel labels for each vertex
        2. Uses geodesic propagation to spread labels across the surface
        3. Assigns face labels via majority voting
        4. Exports each segment as a separate mesh
    
    Args:
        mesh: Input mesh (Trimesh or Scene)
        label_to_points (dict): Maps label (str) -> voxel coordinates (N, 3)
        out_dir (str): Output directory for segmented meshes
        seed_tau_ratio (float): Ratio of bbox diagonal for seed threshold
        min_seed_faces (int): Minimum faces per label (for empty label handling)
    """
    # Handle scene objects (merge all geometries)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])

    # Compute seed threshold based on mesh size
    V = mesh.vertices
    bbox_diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    tau_seed = bbox_diag * seed_tau_ratio

    print(f"Mesh bounding box diagonal: {bbox_diag:.4f}")
    print(f"Seed distance threshold: {tau_seed:.4f}")

    # Step 1: Find nearest label for each vertex
    print("Computing nearest labels for vertices...")
    nearest_lab, dmin, _ = nearest_label_all_vertices(mesh.vertices, label_to_points)

    # Step 2: Build edge graph and propagate labels geodesically
    print("Building edge graph...")
    neighbors, weights = build_edge_graph(mesh)
    
    print("Propagating labels via geodesic distance...")
    seed_mask = (dmin <= tau_seed)
    vlabels, vdist = multisource_geodesic_propagation_with_fallback(
        neighbors, weights,
        seed_mask=seed_mask,
        seed_labels=nearest_lab,
        fallback_labels=nearest_lab
    )

    # Step 3: Assign face labels by majority voting
    print("Assigning face labels...")
    flabels = face_majority_label(mesh, vlabels, vdist)

    # Step 4: Ensure all labels have faces
    flabels = ensure_nonempty_per_label(mesh, flabels, label_to_points, min_faces=min_seed_faces)

    # Step 5: Export submeshes
    print("Exporting segmented meshes...")
    export_label_submeshes(mesh, flabels, out_dir)


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

    # File handler
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Parse Arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Segment 3D meshes into parts based on VLM-generated voxel labels"
    )
    parser.add_argument(
        "--index", type=int, default=0,
        help="Process index (for parallel processing)"
    )
    parser.add_argument(
        "--range", type=int, default=2000,
        help="Processing range (unused, kept for compatibility)"
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    basepath = './test_demo'
    namelist = os.listdir(basepath)
    
    # Setup logging
    logger = get_logger(
        os.path.join(f'exp_split{args.index}.log'),
        verbosity=1
    )
    logger.info('Starting mesh segmentation...')
    logger.info(f'Found {len(namelist)} items to process')

    # -------------------------------------------------------------------------
    # Process Each Item
    # -------------------------------------------------------------------------
    VOXEL_GRID_SIZE = 32  # Size of the voxel grid from VLM

    for name in namelist:
        tmpdir = os.path.join(basepath, name)
        glb_path = os.path.join(tmpdir, 'sample.glb')
        
        if os.path.exists(glb_path):
            logger.info(f'Processing: {name}')
            
            # Create output directories
            os.makedirs(tmpdir, exist_ok=True)
            os.makedirs(os.path.join(tmpdir, 'objs'), exist_ok=True)
            
            # Load mesh and apply rotation (GLB uses different coordinate system)
            mesh = trimesh.load(glb_path, force='mesh')
            R = trimesh.transformations.rotation_matrix(np.deg2rad(90), [1, 0, 0])
            mesh.apply_transform(R)
            
            # Load voxel labels for each part
            loaded = {}
            index = 0
            
            while os.path.exists(os.path.join(tmpdir, f'ind_{index}.npy')):
                # Load voxel coordinates and normalize to [-0.5, 0.5] range
                vertices = np.load(os.path.join(tmpdir, f'ind_{index}.npy'))
                vertices = vertices / VOXEL_GRID_SIZE - 0.5
                loaded[str(index)] = vertices
                index += 1
            
            logger.info(f'  Loaded {len(loaded)} part labels')
            
            # Run segmentation
            segment_mesh_by_wrapped_pcd_no_minus1(
                mesh=mesh,
                label_to_points=loaded,
                out_dir=os.path.join(tmpdir, 'objs'),
                seed_tau_ratio=0.02,    # 2% of bbox diagonal
                min_seed_faces=20       # Minimum faces per segment
            )
            
            logger.info(f'Complete: {name}')
        else:
            logger.info(f'Skip (no GLB): {name}')

    logger.info('All processing complete!')

                
