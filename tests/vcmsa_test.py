#!/usr/bin/env python
from __future__ import print_function
import vcmsa
import unittest
try:
    import numpy as np
    from vcmsa.vcmsa_ph_clustering import (
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

    #def test1(self):
    #    '''
    #    Test of summary
    #    '''
    #    pp = vcmsa.PassageParser()
    #    p = pp.parse_passage("E1_E3")
    #    self.assertTrue(p.summary, list)
    #
    #def test2(self):
    #    '''
    #    Test example case
    #    '''

    #    pp = vcmsa.PassageParser()
    #    p = pp.parse_passage("Mdcksiat2_E3", 3)
    #    #print(vars(p))
    #    self.assertTrue(p.original == "Mdcksiat2_E3")
    #    self.assertTrue(p.plain_format == "MDCKSIAT2_E3")
    #    self.assertTrue(p.coerced_format == "S2_E3")
    #    self.assertTrue(p.ordered_passages) == ['MDCKSIAT2', 'E3']
    #    self.assertTrue(p.min_passages == 5)
    #    self.assertTrue(p.total_passages == 5)
    #    self.assertTrue(p.nth_passage == 'EGG')
    #    self.assertTrue(p.general_passages== ["CANINECELL", "EGG"])
    #    self.assertTrue(p.specific_passages == ["SIAT", "EGG"])
    #    self.assertTrue(p.passage_series == [[1, 'SIAT'], [2, 'SIAT'], [3, 'EGG'], [4, 'EGG'], [5, 'EGG']]) 
    #    self.assertTrue(p.summary == ['Mdcksiat2_E3', 'MDCKSIAT2_E3', 'S2_E3', 'CANINECELL+EGG', 'SIAT+EGG', 'exactly', '5'])
    #         

    #def test3(self):
    #    '''
    #    Test an empty passage annotation
    #    '''
    #    pp = vcmsa.PassageParser()
    #    p = pp.parse_passage("")
    #    self.assertTrue(p.original == "")
    #    self.assertTrue(p.plain_format == "")
    #    self.assertTrue(p.coerced_format == "")
    #    self.assertTrue(p.summary == ['','','','','','',''])

    #    self.assertTrue(p.min_passages == "")
    #    self.assertTrue(p.total_passages == "")
    #    self.assertTrue(p.nth_passage == "")
    #    self.assertTrue(p.general_passages== [])
    #    self.assertTrue(p.specific_passages == [])
    #    self.assertTrue(p.passage_series == []) 
    #         


    #def test4(self):
    #    '''
    #    Check a a longer list of passage IDs
    #    and write an outfile of the summary test
    #    These passage IDs are already partially formatted
    #    '''        

    #    with open("tests/test_passageIDs1.txt", "r") as passageIDs:
    #        with open("tests/output_test_passageIDs1.txt", "w") as outfile:
    #            for ID in passageIDs.readlines():
    #                pp = vcmsa.PassageParser()
    #                input_ID = ID.replace("\n", "") 
    #                full_annotation = pp.parse_passage(input_ID)
    #                quick_annotation = full_annotation.summary
    #                outfile.write(",".join(quick_annotation) + "\n")

    #def test5(self):
    #    '''
    #    Check another list of passage IDs
    #    and write an outfile
    #    '''
    #    with open("tests/test_passageIDs2.txt", "r") as passageIDs:
    #        with open("tests/output_test_passageIDs2.txt", "w") as outfile:
    #            for ID in passageIDs.readlines():
    #                pp = vcmsa.PassageParser()
    #                quick_annotation = pp.parse_passage(ID).summary
    #                outfile.write(str(",".join(quick_annotation)) + "\n")

    #def test6(self):
    #    '''
    #    Test a nonsense passage annotation
    #    '''
    #    pp = vcmsa.PassageParser()
    #    p = pp.parse_passage("asdk?&~EE8")
    #    self.assertTrue(p.original == "asdk?&~EE8")
    #    self.assertTrue(p.plain_format == "ASDK_EE8")
    #    self.assertTrue(p.coerced_format == "")
    #    self.assertTrue(p.summary == ['asdk?&~EE8', 'ASDK_EE8', '', '', '', '', ''])

    #    self.assertTrue(p.min_passages == "")
    #    self.assertTrue(p.total_passages == "")
    #    self.assertTrue(p.nth_passage == "")
    #    self.assertTrue(p.general_passages== [])
    #    self.assertTrue(p.specific_passages == [])
    #    self.assertTrue(p.passage_series == []) 
    

if __name__ == "__main__":
    pass 
