import logging
import numpy as np
import torch
from functools import lru_cache
from pathlib import Path
from scipy.spatial.distance import pdist, squareform, cosine
from sklearn.cluster import DBSCAN
from transformers import AutoTokenizer, EsmModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ESMC_600M support
# ---------------------------------------------------------------------------

_ESMC_MODEL_CACHE = {}


def _remap_esmc_key(key):
    """Remap safetensors weight keys to ESMC model state_dict keys."""
    if "_extra_state" in key:
        return None
    if key.startswith("esmc."):
        key = key[len("esmc."):]
    if key.startswith("lm_head."):
        key = "sequence_head." + key[len("lm_head."):]
    key = key.replace(".ffn.layer_norm_weight", ".ffn.0.weight")
    key = key.replace(".ffn.layer_norm_bias", ".ffn.0.bias")
    key = key.replace(".ffn.fc1_weight", ".ffn.1.weight")
    key = key.replace(".ffn.fc2_weight", ".ffn.3.weight")
    key = key.replace(".attn.layernorm_qkv.layer_norm_weight", ".attn.layernorm_qkv.0.weight")
    key = key.replace(".attn.layernorm_qkv.layer_norm_bias", ".attn.layernorm_qkv.0.bias")
    key = key.replace(".attn.layernorm_qkv.weight", ".attn.layernorm_qkv.1.weight")
    return key


def _get_esmc_tokenizer():
    """Create ESMC tokenizer with compatibility handling.

    Works around the cls_token AttributeError that occurs with certain
    transformers + esm version combinations where the special token
    property setters are removed or incompatible.
    """
    try:
        from esm.tokenization import get_esmc_model_tokenizers
        return get_esmc_model_tokenizers()
    except (AttributeError, TypeError) as e:
        logger.warning("get_esmc_model_tokenizers() failed (%s), using fallback", e)
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

        # The root cause: SpecialTokensMixin defines cls_token etc. as
        # read-only properties (no setter) in certain transformers versions,
        # but __init__ tries to setattr them. Fix: temporarily inject setters
        # that write to the expected backing attribute (_cls_token, etc.).
        _SPECIAL_ATTRS = [
            "bos_token", "eos_token", "unk_token", "sep_token",
            "pad_token", "cls_token", "mask_token", "additional_special_tokens",
        ]

        patches = {}  # (class, attr_name) -> original_descriptor
        for attr_name in _SPECIAL_ATTRS:
            for klass in EsmSequenceTokenizer.__mro__:
                if attr_name not in klass.__dict__:
                    continue
                desc = klass.__dict__[attr_name]
                if isinstance(desc, property) and desc.fset is None:
                    backing = "_" + attr_name

                    def _make_setter(bk):
                        def fset(self, value):
                            object.__setattr__(self, bk, value)
                        return fset

                    new_prop = property(desc.fget, _make_setter(backing), desc.fdel, desc.__doc__)
                    patches[(klass, attr_name)] = desc
                    setattr(klass, attr_name, new_prop)
                break

        try:
            tokenizer = EsmSequenceTokenizer()
        finally:
            # Restore original read-only properties
            for (klass, attr_name), orig_desc in patches.items():
                setattr(klass, attr_name, orig_desc)

        # The esm package's _get_token() calls self.__getattr__(name) directly,
        # which requires __getattr__ to be defined as a method. In transformers
        # 4.44 it may not exist. Add a fallback that reads the backing attrs.
        if not hasattr(tokenizer, "__getattr__"):
            def _fallback_getattr(self, name):
                backing = "_" + name
                try:
                    return object.__getattribute__(self, backing)
                except AttributeError:
                    raise AttributeError(
                        "'{}' object has no attribute '{}'".format(type(self).__name__, name)
                    )
            EsmSequenceTokenizer.__getattr__ = _fallback_getattr

        return tokenizer


def _load_esmc_model(model_path, device="cpu", use_flash_attn=True):
    """Load ESMC_600M model from a local directory or via from_pretrained.

    Parameters
    ----------
    model_path : str
        Either a local directory containing *.safetensors files, or
        a pretrained identifier like "esmc_600m".
    device : str
        Target device ("cpu" or "cuda").
    use_flash_attn : bool
        Whether to enable flash attention.

    Returns
    -------
    model : ESMC model instance (eval mode, on device).
    """
    cache_key = (model_path, device)
    if cache_key in _ESMC_MODEL_CACHE:
        return _ESMC_MODEL_CACHE[cache_key]

    from esm.models.esmc import ESMC

    local_dir = Path(model_path)
    if local_dir.is_dir():
        # Local loading with key remapping
        from safetensors.torch import load_file

        logger.info("Loading ESMC_600M from local path: %s", model_path)
        tokenizer = _get_esmc_tokenizer()
        model = ESMC(
            d_model=1152, n_heads=18, n_layers=36,
            tokenizer=tokenizer,
            use_flash_attn=use_flash_attn,
        ).eval()

        raw_state_dict = {}
        for f in sorted(local_dir.glob("*.safetensors")):
            logger.debug("  Loading %s", f.name)
            raw_state_dict.update(load_file(f, device=device))

        remapped = {}
        for k, v in raw_state_dict.items():
            new_key = _remap_esmc_key(k)
            if new_key is not None:
                remapped[new_key] = v

        model.load_state_dict(remapped, strict=True)
        model = model.to(device)
    else:
        # Use from_pretrained (registers local model or downloads)
        logger.info("Loading ESMC model via from_pretrained: %s", model_path)
        import esm.pretrained as esm_pretrained
        from esm.utils.constants.models import ESMC_600M as ESMC_600M_CONST

        # Register a factory that disables flash_attn if device is cpu
        def _factory(device=device, use_flash_attn=use_flash_attn):
            tokenizer = _get_esmc_tokenizer()
            m = ESMC(
                d_model=1152, n_heads=18, n_layers=36,
                tokenizer=tokenizer,
                use_flash_attn=use_flash_attn,
            ).eval()
            return m.to(device)

        if ESMC_600M_CONST not in esm_pretrained.LOCAL_MODEL_REGISTRY:
            esm_pretrained.LOCAL_MODEL_REGISTRY[ESMC_600M_CONST] = _factory

        model = ESMC.from_pretrained(model_path).to(device)

    _ESMC_MODEL_CACHE[cache_key] = model
    logger.info("ESMC model loaded on %s", device)
    return model


def get_esmc_hidden_states(input_sequence, model_path="esmc_600m", layer=-1, device=None):
    """Extract hidden states from ESMC_600M for a single protein sequence.

    Returns per-residue embeddings from the specified layer, with special
    tokens removed. Shape: (seq_len, embedding_dim)

    Parameters
    ----------
    input_sequence : str
        Amino acid sequence.
    model_path : str
        Path to local ESMC model directory, or "esmc_600m" for pretrained.
    layer : int
        Which hidden layer to extract. -1 = last layer, 0-indexed otherwise.
        ESMC_600M has 36 layers (indices 0..35).
    device : torch.device or None
        Target device. Auto-detects GPU if None.

    Returns
    -------
    ndarray of shape (seq_len, embedding_dim)
        Per-residue hidden states.
    """
    from esm.sdk.api import ESMProtein, ESMProteinError, LogitsConfig

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = str(device)

    # Disable flash attention on CPU
    use_flash_attn = (device_str != "cpu")
    model = _load_esmc_model(model_path, device=device_str, use_flash_attn=use_flash_attn)

    embedding_config = LogitsConfig(sequence=True, return_hidden_states=True)

    protein = ESMProtein(sequence=input_sequence)
    protein_tensor = model.encode(protein)
    output = model.logits(protein_tensor, embedding_config)
    if isinstance(output, ESMProteinError):
        raise RuntimeError("ESMC inference failed: {}".format(output))

    # output.hidden_states shape: (n_layers, 1, L, D)
    # Remove batch dim and special tokens (first and last)
    hs = output.hidden_states  # (n_layers, 1, L, D)
    hs = hs[:, 0, 1:-1, :]    # (n_layers, seq_len, D)

    n_layers = hs.shape[0]
    if layer == -1 or layer is None:
        layer_idx = n_layers - 1
    elif layer < 0:
        layer_idx = n_layers + layer
    else:
        layer_idx = layer

    if layer_idx < 0 or layer_idx >= n_layers:
        raise ValueError(
            "Invalid layer index {} for ESMC model with {} layers".format(layer, n_layers)
        )

    hidden_states = hs[layer_idx].float().detach().cpu().numpy()  # (seq_len, D)
    return hidden_states


def _safe_diagram(diagram):
    """Ensure a persistence diagram is a well-formed (N, 2) float array."""
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
    """Load and cache ESM-2 model and tokenizer."""
    logger.info("Loading ESM-2 model: %s on %s", model_name, device_type)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(torch.device(device_type))
    model.eval()
    return tokenizer, model


def get_esm2_hidden_states(input_sequence, model_name="facebook/esm2_t33_650M_UR50D", layer=-1, device=None):
    """Extract hidden states from ESM-2 for a single protein sequence.

    Returns per-residue embeddings with [CLS] and [EOS] tokens removed.
    Shape: (seq_len, embedding_dim)
    """
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

    # Remove [CLS] (index 0) and [EOS] (index -1) tokens
    hidden_states = outputs.hidden_states[layer_idx][0, 1:-1, :].detach().cpu().numpy()
    return hidden_states


def compute_persistent_homology(hidden_states, max_dimension=3, dimensions=None):
    """Compute persistent homology from residue-level hidden states.

    Per the HF blog algorithm:
    1. Compute pairwise Euclidean distance matrix between residue embeddings
    2. Build Rips complex from the distance matrix
    3. Compute persistence and extract persistence diagrams

    Parameters
    ----------
    hidden_states : ndarray of shape (seq_len, embedding_dim)
        Per-residue embeddings for one protein.
    max_dimension : int
        Maximum simplex dimension for Rips complex construction.
    dimensions : list of int or None
        Which homology dimensions to return diagrams for.
        Default: [0, 1] (connected components and loops).

    Returns
    -------
    dict mapping dimension (int) -> persistence diagram (ndarray of shape (N, 2))
    """
    try:
        import gudhi as gd
    except ImportError as exc:
        raise ImportError("gudhi is required for persistent homology computation") from exc

    if dimensions is None:
        dimensions = [0, 1]

    empty_result = {dim: np.empty((0, 2), dtype=float) for dim in dimensions}

    if hidden_states is None or len(hidden_states) == 0:
        return empty_result
    if len(hidden_states) == 1:
        return {dim: np.array([[0.0, 0.0]], dtype=float) for dim in dimensions}

    pairwise_distances = pdist(hidden_states, metric="euclidean")
    distance_matrix = squareform(pairwise_distances)
    max_edge = float(np.max(distance_matrix))
    rips_complex = gd.RipsComplex(distance_matrix=distance_matrix, max_edge_length=max_edge)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    simplex_tree.persistence()

    result = {}
    for dim in dimensions:
        intervals = simplex_tree.persistence_intervals_in_dimension(dim)
        result[dim] = _safe_diagram(intervals)

    return result


def compute_wasserstein_distance_matrix(persistent_diagrams, order=1.0, dimensions=None):
    """Compute pairwise Wasserstein distances between persistence diagrams.

    Per the HF blog, distances are computed separately for each homology
    dimension and then summed to produce the final distance matrix.

    Parameters
    ----------
    persistent_diagrams : list of dict or list of ndarray
        If dicts: each maps dimension -> diagram array.
        If ndarrays (legacy): treated as dimension-0 diagrams only.
    order : float
        Order of the Wasserstein distance (default 1.0).
    dimensions : list of int or None
        Which dimensions to include in the combined distance.
        Default: inferred from the first diagram's keys, or [0] for legacy format.

    Returns
    -------
    ndarray of shape (num_sequences, num_sequences) with combined Wasserstein distances.
    """
    try:
        from gudhi.hera import wasserstein_distance
    except ImportError as exc:
        raise ImportError("gudhi[hera] is required for Wasserstein distance computation") from exc

    num_diagrams = len(persistent_diagrams)
    if num_diagrams == 0:
        return np.zeros((0, 0), dtype=float)

    # Handle legacy format (list of ndarrays = dimension 0 only)
    is_legacy = not isinstance(persistent_diagrams[0], dict)
    if is_legacy:
        persistent_diagrams = [{0: d} for d in persistent_diagrams]

    if dimensions is None:
        dimensions = sorted(persistent_diagrams[0].keys())

    wasserstein_distances = np.zeros((num_diagrams, num_diagrams), dtype=float)

    for dim in dimensions:
        logger.debug("Computing Wasserstein distances for dimension %d", dim)
        for i in range(num_diagrams):
            diag_i = _safe_diagram(persistent_diagrams[i].get(dim))
            for j in range(i + 1, num_diagrams):
                diag_j = _safe_diagram(persistent_diagrams[j].get(dim))
                if len(diag_i) == 0 and len(diag_j) == 0:
                    distance = 0.0
                else:
                    distance = float(wasserstein_distance(diag_i, diag_j, order=order))
                wasserstein_distances[i, j] += distance
                wasserstein_distances[j, i] += distance

    return wasserstein_distances


def cluster_by_persistent_homology(wasserstein_distance_matrix, eps=0.5, min_samples=2):
    """Cluster sequences using DBSCAN on the Wasserstein distance matrix.

    Parameters
    ----------
    wasserstein_distance_matrix : ndarray of shape (N, N)
    eps : float
        DBSCAN neighborhood radius.
    min_samples : int
        Minimum samples in a neighborhood.

    Returns
    -------
    cluster_labels : ndarray of shape (N,)
        Cluster label for each sequence (-1 = noise).
    clusters : dict mapping label -> list of indices
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    cluster_labels = dbscan.fit_predict(wasserstein_distance_matrix)

    clusters = {}
    for idx, label in enumerate(cluster_labels):
        clusters.setdefault(int(label), []).append(idx)

    return cluster_labels, clusters


def convert_ph_clusters_to_vcmsa_format(cluster_labels, seqs, seq_names, hidden_states_list):
    """Convert PH cluster labels to vcMSA-compatible format.

    Noise points (label=-1) are each placed in their own singleton cluster
    so they can still be processed by the downstream alignment.

    Returns
    -------
    cluster_seqnums_list : list of list of int
    cluster_seqs_list : list of list of str
    cluster_names_list : list of list of str
    """
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
    """Pad hidden states within each cluster to uniform length for do_msa.

    Returns a list of ndarrays, each of shape (num_seqs_in_cluster, max_len, emb_dim).
    """
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
    """Identify reciprocal best hits (RBH) within each cluster.

    Uses mean-pooled hidden states and cosine similarity to find
    bidirectional best matches within clusters.

    Returns
    -------
    list of (name_i, name_j, score) tuples
    """
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


def run_ph_pipeline(seqs, seq_names, esm_model="facebook/esm2_t33_650M_UR50D",
                    ph_layer=-1, ph_eps=0.5, ph_min_samples=2,
                    ph_dimensions=None, cpu_only=False, esm_backend="esm2"):
    """Run the complete persistent homology clustering pipeline.

    This is a convenience function that chains all PH steps:
    1. Extract hidden states (ESM-2 or ESMC)
    2. Compute persistent homology (multi-dimensional)
    3. Compute Wasserstein distance matrix
    4. Cluster with DBSCAN
    5. Convert to vcMSA format

    Parameters
    ----------
    seqs : list of str
        Protein sequences.
    seq_names : list of str
        Sequence identifiers.
    esm_model : str
        Model identifier. For esm2 backend: HuggingFace model name.
        For esmc backend: local path or "esmc_600m".
    ph_layer : int
        Which hidden layer to extract (-1 = last).
    ph_eps : float
        DBSCAN eps parameter.
    ph_min_samples : int
        DBSCAN min_samples parameter.
    ph_dimensions : list of int or None
        Homology dimensions for persistence (default [0, 1]).
    cpu_only : bool
        Force CPU even if GPU is available.
    esm_backend : str
        Which model backend to use: "esm2" (default) or "esmc".

    Returns
    -------
    dict with keys:
        cluster_seqnums_list, cluster_seqs_list, cluster_names_list,
        cluster_hstates_list, hidden_states_list, rbh_pairs, to_exclude
    """
    if ph_dimensions is None:
        ph_dimensions = [0, 1]

    ph_device = torch.device("cuda" if torch.cuda.is_available() and not cpu_only else "cpu")

    # Step 1: Extract hidden states
    backend_label = esm_backend.upper()
    logger.info("PH Step 1: Extracting %s hidden states for %d sequences", backend_label, len(seqs))
    hidden_states_list = []
    for idx, seq in enumerate(seqs):
        if esm_backend == "esmc":
            hs = get_esmc_hidden_states(seq, model_path=esm_model, layer=ph_layer, device=ph_device)
        else:
            hs = get_esm2_hidden_states(seq, model_name=esm_model, layer=ph_layer, device=ph_device)
        hidden_states_list.append(hs)
        if (idx + 1) % 10 == 0 or idx == len(seqs) - 1:
            logger.debug("  Processed %d/%d sequences", idx + 1, len(seqs))

    # Step 2: Compute persistent homology
    logger.info("PH Step 2: Computing persistent homology (dimensions %s)", ph_dimensions)
    persistent_diagrams = []
    for idx, hs in enumerate(hidden_states_list):
        pd = compute_persistent_homology(hs, dimensions=ph_dimensions)
        persistent_diagrams.append(pd)
        if (idx + 1) % 10 == 0 or idx == len(seqs) - 1:
            logger.debug("  Computed PH for %d/%d sequences", idx + 1, len(seqs))

    # Step 3: Compute Wasserstein distance matrix
    logger.info("PH Step 3: Computing Wasserstein distance matrix")
    wasserstein_matrix = compute_wasserstein_distance_matrix(
        persistent_diagrams, dimensions=ph_dimensions
    )

    # Step 4: Cluster with DBSCAN
    logger.info("PH Step 4: Clustering with DBSCAN (eps=%.3f, min_samples=%d)", ph_eps, ph_min_samples)
    cluster_labels, _ = cluster_by_persistent_homology(
        wasserstein_matrix, eps=ph_eps, min_samples=ph_min_samples
    )
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = int(np.sum(cluster_labels == -1))
    logger.info("  Found %d clusters and %d noise points", n_clusters, n_noise)

    # Step 5: Convert to vcMSA format
    cluster_seqnums_list, cluster_seqs_list, cluster_names_list = \
        convert_ph_clusters_to_vcmsa_format(cluster_labels, seqs, seq_names, hidden_states_list)
    cluster_hstates_list = build_cluster_hidden_states_for_vcmsa(
        cluster_seqnums_list, hidden_states_list
    )

    # Step 6: Identify RBH pairs
    rbh_pairs = identify_rbh_in_clusters(
        cluster_labels, hidden_states_list, seq_names=seq_names
    )
    logger.info("PH pipeline complete: %d clusters, %d RBH pairs", len(cluster_seqnums_list), len(rbh_pairs))

    return {
        "cluster_seqnums_list": cluster_seqnums_list,
        "cluster_seqs_list": cluster_seqs_list,
        "cluster_names_list": cluster_names_list,
        "cluster_hstates_list": cluster_hstates_list,
        "hidden_states_list": hidden_states_list,
        "rbh_pairs": rbh_pairs,
        "to_exclude": [],
    }
