from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components, shortest_path


@dataclass
class BoundarySplitResult:
    corners: Dict[str, int]
    edge_tables: Dict[str, pd.DataFrame]
    boundary_table: pd.DataFrame
    graph_radius: float
    median_nn_distance: float


def load_all_nodes(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["node_id", "X_m", "Y_m", "Z_m"],
        engine="python",
    )
    df["node_id"] = df["node_id"].astype(int)
    for col in ["X_m", "Y_m", "Z_m"]:
        df[col] = df[col].astype(float)
    if not df["node_id"].is_unique:
        raise ValueError("node_id in shifted_all_nodes.txt is not unique.")
    return df


def parse_ansys_temperature_listing(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")


    pairs = re.findall(r"^\s*(\d+)\s+([-+]seconds\d+(seconds:\.\d+)seconds)\s*$", text, flags=re.MULTILINE)
    if not pairs:
        raise ValueError("Failed to extract any NODE-TEMP data from final_ansys_temperatures_only.txt.")

    df = pd.DataFrame(pairs, columns=["node_id", "Temperature_C"])
    df["node_id"] = df["node_id"].astype(int)
    df["Temperature_C"] = df["Temperature_C"].astype(float)

    if not df["node_id"].is_unique:
        dup = df.loc[df["node_id"].duplicated(), "node_id"].tolist()[:10]
        raise ValueError(f"Duplicate node_id values found in the temperature table; example: {dup}")
    return df


def build_full_field_truth(nodes_path: str | Path, temps_path: str | Path) -> pd.DataFrame:
    nodes_df = load_all_nodes(nodes_path)
    temps_df = parse_ansys_temperature_listing(temps_path)

    full_df = nodes_df.merge(temps_df, on="node_id", how="outer", indicator=True)
    merge_state = full_df["_merge"].value_counts().to_dict()
    if set(full_df["_merge"].unique()) != {"both"}:
        raise ValueError(f"Nodes and temperatures are not fully aligned; merge state: {merge_state}")

    full_df = full_df.drop(columns=["_merge"]).sort_values("node_id").reset_index(drop=True)
    return full_df


def attach_boundary_node_ids(boundary_df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:

    key_cols = ["X_m", "Y_m", "Z_m"]
    merged = boundary_df.merge(full_df[["node_id", *key_cols]], on=key_cols, how="left")
    if merged["node_id"].isna().any():
        raise ValueError("Some boundary points could not be matched back to full-field node_id.")
    merged["node_id"] = merged["node_id"].astype(int)
    return merged


def build_boundary_graph(points: np.ndarray, radius_scale_candidates: Iterable[float] = (1.05, 1.10, 1.20, 1.30, 1.50)):
    tree = cKDTree(points)
    nn_dist, _ = tree.query(points, k=2)
    median_nn = float(np.median(nn_dist[:, 1]))

    for scale in radius_scale_candidates:
        radius = scale * median_nn
        adjacency = lil_matrix((len(points), len(points)), dtype=float)
        for i, p in enumerate(points):
            nbrs = tree.query_ball_point(p, radius)
            for j in nbrs:
                if i == j:
                    continue
                dist_ij = float(np.linalg.norm(points[i] - points[j]))
                adjacency[i, j] = dist_ij
                adjacency[j, i] = dist_ij
        n_comp, _ = connected_components(adjacency.tocsr(), directed=False)
        if n_comp == 1:
            return adjacency.tocsr(), tree, median_nn, radius

    raise ValueError("The boundary graph could not be connected with the candidate radii; check the boundary data.")


def pick_four_corners_from_extrema(points: np.ndarray) -> Dict[str, int]:
    x = points[:, 0]
    y = points[:, 1]
    bbox_targets = {
        "bottom_left": np.array([x.min(), y.min()]),
        "bottom_right": np.array([x.max(), y.min()]),
        "top_right": np.array([x.max(), y.max()]),
        "top_left": np.array([x.min(), y.max()]),
    }

    available = set(range(len(points)))
    chosen: Dict[str, int] = {}
    for name, target in bbox_targets.items():
        dist2 = np.sum((points - target) ** 2, axis=1)
        for idx in np.argsort(dist2):
            if idx in available:
                chosen[name] = int(idx)
                available.remove(int(idx))
                break
    return chosen


def recover_shortest_path(predecessor_row: np.ndarray, source_idx: int, target_idx: int) -> List[int]:
    path = [int(target_idx)]
    cur = int(target_idx)
    while cur != int(source_idx):
        cur = int(predecessor_row[cur])
        if cur < 0:
            raise ValueError("Shortest-path recovery failed; the graph may be disconnected.")
        path.append(cur)
    path.reverse()
    return path


def point_to_polyline_chainage_and_distance(point: np.ndarray, polyline: np.ndarray) -> Tuple[float, float]:
    seg_lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])

    best_dist = np.inf
    best_chainage = 0.0
    for i in range(len(polyline) - 1):
        a = polyline[i]
        b = polyline[i + 1]
        ab = b - a
        denom = float(np.dot(ab, ab)) + 1e-12
        t = float(np.dot(point - a, ab) / denom)
        t = float(np.clip(t, 0.0, 1.0))
        proj = a + t * ab
        dist = float(np.linalg.norm(point - proj))
        if dist < best_dist:
            best_dist = dist
            best_chainage = float(cumulative[i] + t * seg_lengths[i])
    return best_chainage, best_dist


def split_boundary_into_four_edges(boundary_df: pd.DataFrame) -> BoundarySplitResult:
    work_df = boundary_df.copy().reset_index(drop=True)
    points = work_df[["X_m", "Y_m"]].to_numpy(dtype=float)

    graph, tree, median_nn, graph_radius = build_boundary_graph(points)
    corners = pick_four_corners_from_extrema(points)

    cycle_names = ["bottom_left", "bottom_right", "top_right", "top_left", "bottom_left"]
    edge_name_map = {
        ("bottom_left", "bottom_right"): "Bottom",
        ("bottom_right", "top_right"): "Right",
        ("top_right", "top_left"): "Top",
        ("top_left", "bottom_left"): "Left",
    }

    corner_idx_list = [corners[name] for name in cycle_names[:-1]]
    dist_mat, pred_mat = shortest_path(graph, directed=False, indices=corner_idx_list, return_predecessors=True)

    edge_paths: Dict[str, List[int]] = {}
    covered: set[int] = set()
    for src_name, dst_name in zip(cycle_names[:-1], cycle_names[1:]):
        src_pos = cycle_names[:-1].index(src_name)
        src_idx = corners[src_name]
        dst_idx = corners[dst_name]
        path = recover_shortest_path(pred_mat[src_pos], src_idx, dst_idx)
        edge_name = edge_name_map[(src_name, dst_name)]
        edge_paths[edge_name] = path
        covered.update(path)


    unassigned = sorted(set(range(len(work_df))) - covered)
    polyline_map = {name: points[idx_list] for name, idx_list in edge_paths.items()}

    segment_assignments: Dict[str, List[int]] = {name: list(idx_list) for name, idx_list in edge_paths.items()}
    for idx in unassigned:
        p = points[idx]
        best_edge = None
        best_dist = np.inf
        for edge_name, polyline in polyline_map.items():
            _, dist = point_to_polyline_chainage_and_distance(p, polyline)
            if dist < best_dist:
                best_dist = dist
                best_edge = edge_name
        assert best_edge is not None
        segment_assignments[best_edge].append(idx)


    edge_tables: Dict[str, pd.DataFrame] = {}
    for edge_name, idx_list in segment_assignments.items():
        ref_polyline = polyline_map[edge_name]
        rows = []
        for idx in sorted(set(idx_list)):
            chainage, distance_to_ref = point_to_polyline_chainage_and_distance(points[idx], ref_polyline)
            rows.append((idx, chainage, distance_to_ref))
        order_df = pd.DataFrame(rows, columns=["boundary_local_id", "chainage", "distance_to_ref"])
        order_df = order_df.sort_values(["chainage", "distance_to_ref", "boundary_local_id"]).reset_index(drop=True)

        edge_df = work_df.loc[order_df["boundary_local_id"].to_numpy()].copy().reset_index(drop=True)
        edge_df.insert(0, "edge_name", edge_name)
        edge_df.insert(1, "edge_point_id", np.arange(len(edge_df), dtype=int))
        edge_df["chainage"] = order_df["chainage"].to_numpy()
        edge_df["distance_to_ref"] = order_df["distance_to_ref"].to_numpy()
        edge_tables[edge_name] = edge_df


    records = []
    for edge_name, edge_df in edge_tables.items():
        tmp = edge_df.copy()
        tmp["is_corner"] = False
        records.append(tmp)
    boundary_table = pd.concat(records, axis=0, ignore_index=True)

    corner_rev = {v: k for k, v in corners.items()}
    for edge_name, edge_df in edge_tables.items():
        for local_idx in [0, len(edge_df) - 1]:
            node_id = int(edge_df.loc[local_idx, "node_id"])
            mask = (boundary_table["edge_name"] == edge_name) & (boundary_table["node_id"] == node_id)
            boundary_table.loc[mask, "is_corner"] = True
            if node_id in work_df.loc[list(corner_rev.keys()), "node_id"].values:
                pass

    return BoundarySplitResult(
        corners=corners,
        edge_tables=edge_tables,
        boundary_table=boundary_table,
        graph_radius=graph_radius,
        median_nn_distance=median_nn,
    )


def save_boundary_visualization(boundary_df: pd.DataFrame, split_result: BoundarySplitResult, out_path: str | Path) -> None:
    out_path = Path(out_path)
    color_map = {
        "Bottom": "tab:green",
        "Right": "tab:red",
        "Top": "tab:blue",
        "Left": "tab:orange",
    }

    plt.figure(figsize=(11, 5))
    plt.scatter(boundary_df["X_m"], boundary_df["Y_m"], s=6, c="lightgray", label="all boundary points")

    for edge_name in ["Bottom", "Right", "Top", "Left"]:
        edge_df = split_result.edge_tables[edge_name]
        plt.scatter(edge_df["X_m"], edge_df["Y_m"], s=14, c=color_map[edge_name], label=f"{edge_name} ({len(edge_df)})")
        plt.plot(edge_df["X_m"], edge_df["Y_m"], lw=1.1, c=color_map[edge_name])

    corner_name_map = {
        "bottom_left": "BL",
        "bottom_right": "BR",
        "top_right": "TR",
        "top_left": "TL",
    }
    points = boundary_df[["X_m", "Y_m"]].to_numpy()
    for full_name, idx in split_result.corners.items():
        px, py = points[idx]
        plt.scatter([px], [py], s=80, c="black", marker="x")
        plt.text(px, py, f" {corner_name_map[full_name]}", fontsize=10, weight="bold")

    plt.gca().set_aspect("equal")
    plt.xlabel("X / z (m)")
    plt.ylabel("Y / r (m)")
    plt.title("Boundary split check: Bottom / Right / Top / Left")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_temperature_scatter(full_df: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    plt.figure(figsize=(11, 5))
    sc = plt.scatter(full_df["X_m"], full_df["Y_m"], c=full_df["Temperature_C"], s=8, cmap="inferno")
    plt.gca().set_aspect("equal")
    plt.xlabel("X / z (m)")
    plt.ylabel("Y / r (m)")
    plt.title("Full-field ANSYS truth scatter")
    cbar = plt.colorbar(sc)
    cbar.set_label("Temperature (°C)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 + Stage 2 preprocessing for ANSYS -> PhyGeoNet.")
    parser.add_argument("--all-nodes", type=str, default="shifted_all_nodes.txt")
    parser.add_argument("--temperatures", type=str, default="final_ansys_temperatures_only.txt")
    parser.add_argument("--boundary", type=str, default="shifted_boundary_data.csv")
    parser.add_argument("--out-dir", type=str, default="prep_output")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_df = build_full_field_truth(args.all_nodes, args.temperatures)
    full_df.to_csv(out_dir / "full_field_truth.csv", index=False)

    boundary_df = pd.read_csv(args.boundary)
    boundary_df = attach_boundary_node_ids(boundary_df, full_df)
    split_result = split_boundary_into_four_edges(boundary_df)

    boundary_df.to_csv(out_dir / "boundary_points_with_node_id.csv", index=False)
    split_result.boundary_table.to_csv(out_dir / "boundary_points_classified.csv", index=False)
    for edge_name, edge_df in split_result.edge_tables.items():
        edge_df.to_csv(out_dir / f"boundary_{edge_name.lower()}.csv", index=False)

    save_boundary_visualization(boundary_df, split_result, out_dir / "boundary_split_check.png")
    save_temperature_scatter(full_df, out_dir / "full_field_temperature_scatter.png")

    summary = {
        "full_field_node_count": int(len(full_df)),
        "boundary_point_count": int(len(boundary_df)),
        "edge_counts": {k: int(len(v)) for k, v in split_result.edge_tables.items()},
        "corner_points": {
            k: {
                "boundary_local_id": int(v),
                "X_m": float(boundary_df.loc[v, "X_m"]),
                "Y_m": float(boundary_df.loc[v, "Y_m"]),
                "node_id": int(boundary_df.loc[v, "node_id"]),
            }
            for k, v in split_result.corners.items()
        },
        "graph_radius": split_result.graph_radius,
        "median_nearest_neighbor_distance": split_result.median_nn_distance,
    }
    (out_dir / "prep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nDone. Outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
