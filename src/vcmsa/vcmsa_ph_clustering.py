import numpy as np
import torch
from functools import lru_cache
from scipy.spatial.distance import pdist, squareform, cosine
from sklearn.cluster import DBSCAN
from transformers import AutoTokenizer, EsmModel


def _safe_diagram(diagram):
    if diagram is None:
        return np.empty((0, 2), dtype=float)
    arr = np.asarray(diagram, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    finite_mask = np.isfinite(arr).all(axis=1)
    return arr[finite_mask]


@lru_cache(maxsize=2)
def _load_esm2(model_name, device_type):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(torch.device(device_type))
    model.eval()
    return tokenizer, model


def get_esm2_hidden_states(input_sequence, model_name="facebook/esm2_t33_650M_UR50D", layer=-1, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = _load_esm2(model_name, str(device))

    encoded_input = tokenizer([input_sequence], return_tensors="pt", truncation=True)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    with torch.no_grad():
        outputs = model(**encoded_input, output_hidden_states=True)

    num_hidden_layers = model.config.num_hidden_layers
    if layer is None or layer == -1:
        layer_idx = num_hidden_layers
    elif layer < 0:
        layer_idx = (num_hidden_layers + 1) + layer
    else:
        layer_idx = layer

    if layer_idx < 0 or layer_idx >= len(outputs.hidden_states):
        raise ValueError("Invalid hidden-state layer index: {}".format(layer))

    hidden_states = outputs.hidden_states[layer_idx][0, 1:-1, :].detach().cpu().numpy()
    return hidden_states


def compute_persistent_homology(hidden_states, max_dimension=3):
    try:
        import gudhi as gd
    except ImportError as exc:
        raise ImportError("gudhi is required for persistent homology computation") from exc

    if hidden_states is None or len(hidden_states) == 0:
        return np.empty((0, 2), dtype=float)
    if len(hidden_states) == 1:
        return np.array([[0.0, 0.0]], dtype=float)

    pairwise_distances = pdist(hidden_states, metric="euclidean")
    distance_matrix = squareform(pairwise_distances)
    max_edge = float(np.max(distance_matrix))
    rips_complex = gd.RipsComplex(distance_matrix=distance_matrix, max_edge_length=max_edge)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    simplex_tree.persistence()
    persistence_diagram = simplex_tree.persistence_intervals_in_dimension(0)
    return _safe_diagram(persistence_diagram)


def compute_wasserstein_distance_matrix(persistent_diagrams, order=1.0):
    try:
        from gudhi.hera import wasserstein_distance
    except ImportError as exc:
        raise ImportError("gudhi[hera] is required for Wasserstein distance computation") from exc

    num_diagrams = len(persistent_diagrams)
    wasserstein_distances = np.zeros((num_diagrams, num_diagrams), dtype=float)

    for i in range(num_diagrams):
        diag_i = _safe_diagram(persistent_diagrams[i])
        for j in range(i + 1, num_diagrams):
            diag_j = _safe_diagram(persistent_diagrams[j])
            if len(diag_i) == 0 and len(diag_j) == 0:
                distance = 0.0
            else:
                distance = float(wasserstein_distance(diag_i, diag_j, order=order))
            wasserstein_distances[i, j] = distance
            wasserstein_distances[j, i] = distance

    return wasserstein_distances


def cluster_by_persistent_homology(wasserstein_distance_matrix, eps=0.5, min_samples=2):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    cluster_labels = dbscan.fit_predict(wasserstein_distance_matrix)

    clusters = {}
    for idx, label in enumerate(cluster_labels):
        clusters.setdefault(int(label), []).append(idx)

    return cluster_labels, clusters


def convert_ph_clusters_to_vcmsa_format(cluster_labels, seqs, seq_names, hidden_states_list):
    clusters = {}
    for idx, label in enumerate(cluster_labels):
        if label == -1:
            key = "noise_{}".format(idx)
        else:
            key = int(label)
        clusters.setdefault(key, []).append(idx)

    cluster_seqnums_list = []
    cluster_seqs_list = []
    cluster_names_list = []

    for cluster_key in sorted(clusters, key=lambda x: (str(type(x)), x)):
        indices = clusters[cluster_key]
        cluster_seqnums_list.append(indices)
        cluster_seqs_list.append([seqs[i] for i in indices])
        cluster_names_list.append([seq_names[i] for i in indices])

    return cluster_seqnums_list, cluster_seqs_list, cluster_names_list


def build_cluster_hidden_states_for_vcmsa(cluster_seqnums_list, hidden_states_list):
    cluster_hstates_list = []
    for indices in cluster_seqnums_list:
        cluster_hidden_states = []
        for i in indices:
            hs = hidden_states_list[i]
            if hs is None:
                hs = np.zeros((0, 0))
            cluster_hidden_states.append(hs)

        max_len = max([h.shape[0] for h in cluster_hidden_states]) if cluster_hidden_states else 0
        emb_dim = 0
        for hs in cluster_hidden_states:
            if hs is not None and hs.ndim == 2:
                emb_dim = hs.shape[1]
                break
        if cluster_hidden_states and emb_dim == 0:
            raise ValueError(
                "Unable to determine embedding dimension for cluster indices {}: "
                "all hidden states are None or not 2D arrays".format(indices)
            )
        padded = []
        for hs in cluster_hidden_states:
            if hs.ndim != 2:
                hs = np.zeros((0, emb_dim))
            pad_rows = max_len - hs.shape[0]
            if pad_rows > 0:
                hs = np.pad(hs, ((0, pad_rows), (0, 0)))
            padded.append(hs)
        if not padded:
            padded = [np.zeros((0, emb_dim))]
        cluster_hstates_list.append(np.array(padded))
    return cluster_hstates_list


def identify_rbh_in_clusters(cluster_assignments, hidden_states_list, seq_names=None, threshold=0.8):
    if isinstance(cluster_assignments, dict):
        clusters = cluster_assignments
    else:
        clusters = {}
        for idx, label in enumerate(cluster_assignments):
            if int(label) == -1:
                continue
            clusters.setdefault(int(label), []).append(idx)

    sequence_vectors = []
    for hs in hidden_states_list:
        if hs is None or len(hs) == 0:
            sequence_vectors.append(None)
        else:
            sequence_vectors.append(np.mean(hs, axis=0))

    rbh_pairs = []
    for indices in clusters.values():
        if len(indices) < 2:
            continue
        best_match = {}
        best_score = {}
        for i in indices:
            if sequence_vectors[i] is None:
                continue
            scores = []
            for j in indices:
                if i == j or sequence_vectors[j] is None:
                    continue
                score = 1.0 - cosine(sequence_vectors[i], sequence_vectors[j])
                scores.append((j, float(score)))
            if not scores:
                continue
            j_best, score_best = max(scores, key=lambda x: x[1])
            best_match[i] = j_best
            best_score[i] = score_best

        for i, j in best_match.items():
            if best_match.get(j) == i and best_score.get(i, 0.0) >= threshold:
                id_i = seq_names[i] if seq_names else i
                id_j = seq_names[j] if seq_names else j
                pair = tuple(sorted((id_i, id_j)))
                if pair not in {(x[0], x[1]) for x in rbh_pairs}:
                    rbh_pairs.append((pair[0], pair[1], best_score[i]))

    return rbh_pairs
