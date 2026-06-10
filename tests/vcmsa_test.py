#!/usr/bin/env python
from __future__ import print_function
import vcmsa
import unittest
try:
    import numpy as np
    from vcmsa.vcmsa_ph_clustering import (
        _safe_diagram,
        compute_persistent_homology,
        compute_wasserstein_distance_matrix,
        cluster_by_persistent_homology,
        convert_ph_clusters_to_vcmsa_format,
        build_cluster_hidden_states_for_vcmsa,
        identify_rbh_in_clusters,
    )
    HAS_PH_DEPS = True
except ImportError:
    HAS_PH_DEPS = False


class test_vcmsa(unittest.TestCase):


    def test1(self):
        '''
        Fake test
        '''    
        self.assertTrue(True == True)

    @unittest.skipUnless(HAS_PH_DEPS, "PH test dependencies are not installed")
    def test_safe_diagram_edge_cases(self):
        """Test _safe_diagram handles various inputs correctly."""
        # None input
        result = _safe_diagram(None)
        self.assertEqual(result.shape, (0, 2))

        # Empty array
        result = _safe_diagram(np.array([]))
        self.assertEqual(result.shape, (0, 2))

        # 1D input reshaped to Nx2
        result = _safe_diagram(np.array([1.0, 2.0, 3.0, 4.0]))
        self.assertEqual(result.shape, (2, 2))

        # Filters infinite values
        arr = np.array([[1.0, 2.0], [3.0, np.inf], [4.0, 5.0]])
        result = _safe_diagram(arr)
        self.assertEqual(result.shape, (2, 2))

    @unittest.skipUnless(HAS_PH_DEPS, "PH test dependencies are not installed")
    def test_compute_persistent_homology_returns_dict(self):
        """Test that compute_persistent_homology returns multi-dimensional diagrams."""
        hidden_states = np.random.rand(10, 5)
        result = compute_persistent_homology(hidden_states, dimensions=[0, 1])
        self.assertIsInstance(result, dict)
        self.assertIn(0, result)
        self.assertIn(1, result)
        self.assertEqual(result[0].ndim, 2)
        self.assertEqual(result[0].shape[1], 2)

    @unittest.skipUnless(HAS_PH_DEPS, "PH test dependencies are not installed")
    def test_compute_persistent_homology_empty_input(self):
        """Test PH with empty/singleton inputs."""
        # Empty
        result = compute_persistent_homology(np.array([]), dimensions=[0])
        self.assertEqual(result[0].shape, (0, 2))

        # Single residue
        result = compute_persistent_homology(np.array([[1.0, 2.0]]), dimensions=[0])
        self.assertEqual(result[0].shape[0], 1)

    @unittest.skipUnless(HAS_PH_DEPS, "PH test dependencies are not installed")
    def test_wasserstein_distance_matrix_multidim(self):
        """Test Wasserstein distance with multi-dimensional diagrams."""
        diagrams = [
            {0: np.array([[0.0, 1.0], [0.5, 2.0]]), 1: np.array([[0.1, 0.5]])},
            {0: np.array([[0.0, 1.0], [0.5, 2.0]]), 1: np.array([[0.1, 0.5]])},
            {0: np.array([[0.0, 5.0], [1.0, 3.0]]), 1: np.array([[0.0, 2.0]])},
        ]
        matrix = compute_wasserstein_distance_matrix(diagrams, dimensions=[0, 1])
        self.assertEqual(matrix.shape, (3, 3))
        # Identical diagrams should have zero distance
        self.assertAlmostEqual(matrix[0, 1], 0.0, places=5)
        # Different diagrams should have non-zero distance
        self.assertGreater(matrix[0, 2], 0.0)
        # Symmetric
        self.assertAlmostEqual(matrix[0, 2], matrix[2, 0], places=5)

    @unittest.skipUnless(HAS_PH_DEPS, "PH test dependencies are not installed")
    def test_wasserstein_distance_matrix_legacy_format(self):
        """Test Wasserstein distance with legacy ndarray format."""
        diagrams = [
            np.array([[0.0, 1.0], [0.5, 2.0]]),
            np.array([[0.0, 1.0], [0.5, 2.0]]),
            np.array([[0.0, 5.0], [1.0, 3.0]]),
        ]
        matrix = compute_wasserstein_distance_matrix(diagrams)
        self.assertEqual(matrix.shape, (3, 3))
        self.assertAlmostEqual(matrix[0, 1], 0.0, places=5)
        self.assertGreater(matrix[0, 2], 0.0)

    @unittest.skipUnless(HAS_PH_DEPS, "PH test dependencies are not installed")
    def test_ph_cluster_and_convert_format(self):
        wasserstein = np.array([
            [0.0, 0.1, 2.0],
            [0.1, 0.0, 2.0],
            [2.0, 2.0, 0.0],
        ])
        labels, clusters = cluster_by_persistent_homology(wasserstein, eps=0.5, min_samples=2)
        self.assertEqual(labels.tolist(), [0, 0, -1])
        self.assertEqual(clusters[0], [0, 1])

        seqs = ["AAA", "AAT", "GGG"]
        seq_names = ["s1", "s2", "s3"]
        hidden_states = [
            np.array([[1.0, 0.0], [0.9, 0.1], [1.0, 0.0]]),
            np.array([[0.95, 0.05], [0.85, 0.15]]),
            np.array([[0.0, 1.0], [0.1, 0.9], [0.0, 1.0], [0.1, 0.9]]),
        ]
        cluster_seqnums_list, cluster_seqs_list, cluster_names_list = \
            convert_ph_clusters_to_vcmsa_format(labels, seqs, seq_names, hidden_states)
        cluster_hstates_list = build_cluster_hidden_states_for_vcmsa(cluster_seqnums_list, hidden_states)

        self.assertEqual(cluster_seqnums_list, [[0, 1], [2]])
        self.assertEqual(cluster_seqs_list, [["AAA", "AAT"], ["GGG"]])
        self.assertEqual(cluster_names_list, [["s1", "s2"], ["s3"]])
        self.assertEqual(cluster_hstates_list[0].shape, (2, 3, 2))
        self.assertEqual(cluster_hstates_list[1].shape, (1, 4, 2))

    @unittest.skipUnless(HAS_PH_DEPS, "PH test dependencies are not installed")
    def test_identify_rbh_in_clusters(self):
        labels = np.array([0, 0, 0])
        hidden_states = [
            np.array([[1.0, 0.0], [1.0, 0.0]]),
            np.array([[0.99, 0.01], [0.98, 0.02]]),
            np.array([[0.0, 1.0], [0.0, 1.0]]),
        ]
        pairs = identify_rbh_in_clusters(labels, hidden_states, seq_names=["a", "b", "c"], threshold=0.8)
        pair_names = {(x[0], x[1]) for x in pairs}
        self.assertIn(("a", "b"), pair_names)
        self.assertNotIn(("a", "c"), pair_names)
        self.assertNotIn(("b", "c"), pair_names)
        pair_scores = { (x[0], x[1]): x[2] for x in pairs }
        self.assertGreaterEqual(pair_scores[("a", "b")], 0.8)

    @unittest.skipUnless(HAS_PH_DEPS, "PH test dependencies are not installed")
    def test_full_ph_pipeline_integration(self):
        """Integration test: full PH pipeline with synthetic data."""
        # Generate two groups of "sequences" with distinct hidden states
        np.random.seed(42)
        # Group 1: similar hidden states (clustered around [1, 0])
        hs_group1 = [np.random.normal(loc=[1, 0], scale=0.1, size=(5, 2)) for _ in range(3)]
        # Group 2: similar hidden states (clustered around [0, 1])
        hs_group2 = [np.random.normal(loc=[0, 1], scale=0.1, size=(5, 2)) for _ in range(3)]
        all_hs = hs_group1 + hs_group2

        # Compute PH for each
        diagrams = [compute_persistent_homology(hs, dimensions=[0]) for hs in all_hs]

        # Compute Wasserstein matrix
        matrix = compute_wasserstein_distance_matrix(diagrams, dimensions=[0])
        self.assertEqual(matrix.shape, (6, 6))

        # Cluster - with appropriate eps for this synthetic data
        labels, clusters = cluster_by_persistent_homology(matrix, eps=1.0, min_samples=2)

        # Should find at least some clustering structure
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self.assertGreaterEqual(n_clusters, 1)

    @unittest.skipUnless(HAS_PH_DEPS, "PH test dependencies are not installed")
    def test_convert_format_all_noise(self):
        """Test conversion when all points are noise (label=-1)."""
        labels = np.array([-1, -1, -1])
        seqs = ["AA", "BB", "CC"]
        names = ["s1", "s2", "s3"]
        hs = [np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]), np.array([[0.5, 0.5]])]

        seqnums, seqs_out, names_out = convert_ph_clusters_to_vcmsa_format(labels, seqs, names, hs)
        # Each noise point should be in its own cluster
        self.assertEqual(len(seqnums), 3)
        for cluster in seqnums:
            self.assertEqual(len(cluster), 1)


if __name__ == "__main__":
    pass 
