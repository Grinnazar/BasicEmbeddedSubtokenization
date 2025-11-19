"""
Interactive 2D Visualization of Word Embeddings with DBSCAN Clustering.
- 1st Click: Shows nearest neighbors.
- 2nd Click (same point): Highlights all points in the same cluster.
- 3rd Click (same point): Resets the view.
"""

import sys
from pathlib import Path
import numpy as np
import sentencepiece as spm
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.collections
import matplotlib.cm as cm 
import mplcursors

#  Constants
MODEL_PREFIX = "pl_bpe_"
EMBEDDINGS_FILENAME = "embeddings.npy"

# Tsne
NUM_POINTS_TO_VISUALIZE = 1500
TSNE_PERPLEXITY = 5.0
TSNE_MAX_ITER = 3000
TSNE_INIT_METHOD = 'pca'
TSNE_RANDOM_STATE = 322

# Clusters
DBSCAN_EPS = 1.8
DBSCAN_MIN_SAMPLES = 2

# Closest visualization
NUM_NEIGHBORS_TO_SHOW = 30

CLICKED_POINT_COLOR_RGB = np.array([1.0, 0.0, 0.0])  # Red
NEIGHBOR_POINT_COLOR_RGB = np.array([1.0, 0.6, 0.0]) # Orange
CLUSTER_HIGHLIGHT_COLOR_RGB = np.array([0.0, 0.5, 1.0]) # Blue for clusters
NOISE_POINT_COLOR_RGB = np.array([0.5, 0.5, 0.5])    # Grey
INITIAL_ALPHA = 0.7
LOW_ALPHA_OTHERS = 0.2
HIGH_ALPHA_SELECTED = 1.0
HIGH_ALPHA_NEIGHBOR = 0.9
HIGH_ALPHA_CLUSTER = 0.85

class EmbeddingVisualizer:
    def __init__(self, coords_2d, embeddings_subset_hd, labels_subset, cluster_labels, initial_colors_rgba, point_base_rgb_colors):
        # Data
        self.coords_2d = coords_2d
        self.embeddings_subset_hd = embeddings_subset_hd
        self.labels_subset = labels_subset
        self.cluster_labels = cluster_labels
        self.initial_colors_rgba = initial_colors_rgba
        self.point_base_rgb_colors = point_base_rgb_colors
        self.num_to_plot = len(labels_subset)
        
        # State
        self.last_clicked_index = -1
        self.click_mode = 0  # 0: initial, 1: neighbors shown, 2: cluster shown
        self.dynamic_text_labels = []

        # Matplotlib objects
        self.fig, self.ax = plt.subplots(figsize=(14, 12))
        self.scatter = None

    def _clear_dynamic_elements(self):
        """Clears temporary text labels from the plot."""
        for text_obj in self.dynamic_text_labels:
            text_obj.remove()
        self.dynamic_text_labels.clear()

    def _reset_to_initial_state(self):
        """Resets plot to its original state (DBSCAN colors)."""
        if self.scatter is None: return
        
        self.scatter.set_facecolors(self.initial_colors_rgba) 
        self._clear_dynamic_elements()
        self.last_clicked_index = -1
        self.click_mode = 0
        
        print("\n✨ View reset.")
        self.fig.canvas.draw_idle()

    def _highlight_neighbors(self, clicked_idx):
        """Highlights the k-nearest neighbors of a point."""
        print(f"\n👉 Click 1: Showing neighbors for '{self.labels_subset[clicked_idx]}'")
        
        # Find neighbors
        clicked_embedding_hd = self.embeddings_subset_hd[clicked_idx]
        neighbor_indices = find_k_nearest_neighbors(
            clicked_embedding_hd, self.embeddings_subset_hd, k=NUM_NEIGHBORS_TO_SHOW, exclude_index=clicked_idx
        )
        
        # Update colors
        new_colors_rgba = np.zeros((self.num_to_plot, 4))
        for i in range(self.num_to_plot):
            new_colors_rgba[i, :3] = self.point_base_rgb_colors[i]
            new_colors_rgba[i, 3] = LOW_ALPHA_OTHERS

        new_colors_rgba[clicked_idx, :3] = CLICKED_POINT_COLOR_RGB
        new_colors_rgba[clicked_idx, 3] = HIGH_ALPHA_SELECTED

        for neighbor_idx in neighbor_indices:
            new_colors_rgba[neighbor_idx, :3] = NEIGHBOR_POINT_COLOR_RGB
            new_colors_rgba[neighbor_idx, 3] = HIGH_ALPHA_NEIGHBOR

        self.scatter.set_facecolors(new_colors_rgba)
        
        # Update labels
        self._clear_dynamic_elements()
        cx, cy = self.coords_2d[clicked_idx]
        clicked_text = self.ax.text(cx, cy, f" {self.labels_subset[clicked_idx]}", color='black', backgroundcolor=(1,1,1,0.7), fontsize=9, zorder=10, weight='bold')
        self.dynamic_text_labels.append(clicked_text)

        for neighbor_idx in neighbor_indices:
            nx, ny = self.coords_2d[neighbor_idx]
            neighbor_text = self.ax.text(nx, ny, f" {self.labels_subset[neighbor_idx]}", color=NEIGHBOR_POINT_COLOR_RGB, fontsize=8, zorder=9)
            self.dynamic_text_labels.append(neighbor_text)

        self.fig.canvas.draw_idle()

    def _highlight_cluster(self, clicked_idx):
        """Highlights all points belonging to the same cluster."""
        target_cluster_id = self.cluster_labels[clicked_idx]
        print(f"👉 Click 2: Showing all points in cluster {target_cluster_id}")
        
        if target_cluster_id == -1:
            print("   (Point is noise, has no cluster. Resetting.)")
            self._reset_to_initial_state()
            return
            
        # Find all points in the same cluster
        indices_in_cluster = np.where(self.cluster_labels == target_cluster_id)[0]

        # Update colors
        new_colors_rgba = np.zeros((self.num_to_plot, 4))
        for i in range(self.num_to_plot):
            new_colors_rgba[i, :3] = self.point_base_rgb_colors[i]
            new_colors_rgba[i, 3] = LOW_ALPHA_OTHERS
        
        for cluster_member_idx in indices_in_cluster:
            # Use the cluster's original base color but with high alpha
            new_colors_rgba[cluster_member_idx, :3] = self.point_base_rgb_colors[cluster_member_idx]
            new_colors_rgba[cluster_member_idx, 3] = HIGH_ALPHA_CLUSTER
        
        # Make the clicked point stand out
        new_colors_rgba[clicked_idx, :3] = CLICKED_POINT_COLOR_RGB
        new_colors_rgba[clicked_idx, 3] = HIGH_ALPHA_SELECTED

        self.scatter.set_facecolors(new_colors_rgba)
        
        # Update labels
        self._clear_dynamic_elements()
        for cluster_member_idx in indices_in_cluster:
            px, py = self.coords_2d[cluster_member_idx]
            # Use a different color for the main point vs other cluster points
            is_main_point = (cluster_member_idx == clicked_idx)
            color = 'black' if is_main_point else CLUSTER_HIGHLIGHT_COLOR_RGB
            weight = 'bold' if is_main_point else 'normal'
            size = 9 if is_main_point else 8
            
            text = self.ax.text(px, py, f" {self.labels_subset[cluster_member_idx]}", color=color, fontsize=size, weight=weight, zorder=10)
            self.dynamic_text_labels.append(text)
            
        self.fig.canvas.draw_idle()


    def _on_pick(self, event):
        if not isinstance(event.artist, matplotlib.collections.PathCollection) or event.artist != self.scatter: return
        if not len(event.ind): return
        
        clicked_idx = event.ind[0]

        # Case 1: Clicking a NEW point. Always reset state and show neighbors.
        if clicked_idx != self.last_clicked_index:
            self.last_clicked_index = clicked_idx
            self.click_mode = 1
            self._highlight_neighbors(clicked_idx)
        # Case 2: Clicking the SAME point again.
        else:
            if self.click_mode == 1: # From neighbors to cluster
                self.click_mode = 2
                self._highlight_cluster(clicked_idx)
            else: # From cluster (or any other state) to reset
                self._reset_to_initial_state()

    def plot(self, title):
        """Creates and displays the scatter plot."""
        self.scatter = self.ax.scatter(
            self.coords_2d[:, 0], self.coords_2d[:, 1],
            c=self.initial_colors_rgba,
            picker=True,
            pickradius=5
        )
        self.ax.set_xlabel("Dimension 1")
        self.ax.set_ylabel("Dimension 2")
        self.ax.set_title(title)

        self.fig.canvas.mpl_connect('pick_event', self._on_pick)

        cursor = mplcursors.cursor(self.scatter, hover=True)
        cursor.connect("add", lambda sel: sel.annotation.set_text(
            f"{self.labels_subset[sel.index]}\nCluster ID: {self.cluster_labels[sel.index]}"
        ))

        print("✔️ Plot generated. Click a point for neighbors, click again for the cluster, click a third time to reset.")
        print(f"ℹ️  DBSCAN params: eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES}.")
        plt.tight_layout()
        plt.show()


def load_data(model_prefix: str, embeddings_filename: str) -> tuple[spm.SentencePieceProcessor, np.ndarray, list[str]]:
    model_file = Path(f"{model_prefix}.model")
    embeddings_file = Path(embeddings_filename)

    if not model_file.exists():
        print(f"❌ Error: SentencePiece model file not found: {model_file}")
        sys.exit(1)
    if not embeddings_file.exists():
        print(f"❌ Error: Embeddings file not found: {embeddings_file}")
        sys.exit(1)

    print(f"▶️ Loading SentencePiece model from '{model_file}'...")
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_file))
    print(f"✔️ Tokenizer loaded. Vocab size: {sp.get_piece_size()}")

    print(f"▶️ Loading embeddings from '{embeddings_file}'...")
    embeddings = np.load(embeddings_file)
    print(f"✔️ Embeddings loaded. Shape: {embeddings.shape}")

    effective_vocab_limit = min(embeddings.shape[0], sp.get_piece_size())
    if embeddings.shape[0] != sp.get_piece_size():
         print(f"⚠️ Warning: Embeddings rows ({embeddings.shape[0]}) and SP vocab size ({sp.get_piece_size()}) differ. "
               f"Using smallest common count: {effective_vocab_limit}")

    labels = [sp.id_to_piece(i) for i in range(effective_vocab_limit)]
    embeddings = embeddings[:effective_vocab_limit]

    return sp, embeddings, labels

def find_k_nearest_neighbors(target_embedding_hd: np.ndarray,
                             all_embeddings_hd: np.ndarray,
                             k: int,
                             exclude_index: int = -1) -> list[int]:
    if target_embedding_hd.ndim == 2 and target_embedding_hd.shape[0] == 1:
        target_embedding_hd_1d = target_embedding_hd.flatten()
    elif target_embedding_hd.ndim == 1:
        target_embedding_hd_1d = target_embedding_hd
    else:
        raise ValueError("target_embedding_hd must be a 1D array or a 2D array with one row.")

    similarities = cosine_similarity(target_embedding_hd_1d.reshape(1, -1), all_embeddings_hd)[0]
    sorted_indices = np.argsort(similarities)[::-1]

    neighbors_indices = []
    count = 0
    for idx in sorted_indices:
        if idx == exclude_index:
            continue
        if count < k:
            neighbors_indices.append(idx)
            count += 1
        else:
            break
    return neighbors_indices

def main():
    sp, all_embeddings_hd, all_labels = load_data(MODEL_PREFIX, EMBEDDINGS_FILENAME)

    if not all_labels:
        print("❌ No labels to visualize. Exiting.")
        return

    num_to_plot = min(NUM_POINTS_TO_VISUALIZE, len(all_labels))
    if num_to_plot < 1:
        print("❌ Number of points to visualize is zero. Exiting.")
        return

    print(f"\n▶️ Selecting the first {num_to_plot} tokens for visualization.")
    embeddings_subset_hd = all_embeddings_hd[:num_to_plot]
    labels_subset = all_labels[:num_to_plot]

    if embeddings_subset_hd.shape[1] <= 2:
        print(f"▶️ Embeddings are already {embeddings_subset_hd.shape[1]}D or less. Using them directly.")
        coords_2d = np.zeros((num_to_plot, 2))
        coords_2d[:, :embeddings_subset_hd.shape[1]] = embeddings_subset_hd
    else:
        print(f"\n▶️ Performing t-SNE dimensionality reduction from {embeddings_subset_hd.shape[1]}D to 2D...")
        tsne = TSNE(n_components=2, random_state=TSNE_RANDOM_STATE, perplexity=TSNE_PERPLEXITY,
                    max_iter=TSNE_MAX_ITER, init=TSNE_INIT_METHOD, n_jobs=-1, learning_rate='auto')
        coords_2d = tsne.fit_transform(embeddings_subset_hd)
        print("✔️ t-SNE reduction complete.")

    print(f"\n▶️ Performing DBSCAN clustering (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})...")
    dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    cluster_labels = dbscan.fit_predict(coords_2d)

    n_clusters_ = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise_ = list(cluster_labels).count(-1)
    print(f"✔️ DBSCAN complete. Found {n_clusters_} clusters and {n_noise_} noise points.")

    unique_dbscan_labels = sorted(list(set(cluster_labels)))
    point_base_rgb_colors = np.zeros((num_to_plot, 3))

    if n_clusters_ > 0:
        palette = cm.get_cmap('turbo', n_clusters_)(np.linspace(0, 1, n_clusters_))
        cluster_to_palette_idx = {label: i for i, label in enumerate(u for u in unique_dbscan_labels if u != -1)}

        for i, lbl in enumerate(cluster_labels):
            if lbl == -1:
                point_base_rgb_colors[i] = NOISE_POINT_COLOR_RGB
            else:
                point_base_rgb_colors[i] = palette[cluster_to_palette_idx[lbl]][:3]
    else: 
        point_base_rgb_colors[:] = NOISE_POINT_COLOR_RGB

    initial_colors_rgba = np.column_stack((point_base_rgb_colors, np.full(num_to_plot, INITIAL_ALPHA)))
    
    print("\n▶️ Initializing visualizer...")
    visualizer = EmbeddingVisualizer(
        coords_2d=coords_2d,
        embeddings_subset_hd=embeddings_subset_hd,
        labels_subset=labels_subset,
        cluster_labels=cluster_labels,
        initial_colors_rgba=initial_colors_rgba,
        point_base_rgb_colors=point_base_rgb_colors
    )

    plot_title = (f"2D t-SNE & DBSCAN of {num_to_plot} Word Embeddings ({n_clusters_} clusters, {n_noise_} noise)\n"
                  f"1st click: neighbors | 2nd: cluster | 3rd: reset")
                  
    visualizer.plot(title=plot_title)


main()