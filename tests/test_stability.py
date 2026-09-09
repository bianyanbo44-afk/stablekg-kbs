import sys
import unittest
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import synthetic_audit as synthetic
import stability_benchmark as public
import chronological_benchmark as chronological


class StabilityProtocolTests(unittest.TestCase):
    def test_causal_features_exclude_future_evidence(self):
        events = [(2, 10), (8, 20), (12, 10)]
        mass, _, _, _ = public.causal_features(
            public.Query(1, 1, 5, (10,)), events, half_life=35.0, mode="decay"
        )
        self.assertIn(10, mass)
        self.assertNotIn(20, mass)
        self.assertAlmostEqual(sum(mass.values()), mass[10])

    def test_filtered_average_tie_removes_sibling_answers(self):
        mass = {1: 3.0, 2: 3.0, 3: 2.0}
        ranks = public._filtered_ranks({1, 2}, mass, {1, 2}, n_entities=5)
        self.assertEqual(ranks[1], 1.0)
        self.assertEqual(ranks[2], 1.0)

    def test_filtered_zero_tail_has_average_tie_rank(self):
        mass = {1: 2.0}
        ranks = public._filtered_ranks({4}, mass, {4}, n_entities=5)
        # Entity 4 and entities 2, 3, 5 share the zero score.
        self.assertEqual(ranks[4], 3.5)

    def test_multi_answer_mrr_averages_each_answer(self):
        mass = {1: 5.0, 2: 1.0, 3: 4.0, 4: 0.0}
        ranks = public._filtered_ranks({1, 2}, mass, {1, 2}, n_entities=4)
        reciprocal = np.mean([1.0 / ranks[1], 1.0 / ranks[2]])
        self.assertAlmostEqual(reciprocal, (1.0 + 1.0 / 2.0) / 2.0)

    def test_threshold_does_not_depend_on_test_labels(self):
        calibration = [
            {"margin": 0.1, "correct": 0.0},
            {"margin": 0.9, "correct": 1.0},
            {"margin": 0.8, "correct": 1.0},
        ]
        operating = [
            {"belief": 0.2, "correct": 0.0},
            {"belief": 0.8, "correct": 1.0},
            {"belief": 0.7, "correct": 1.0},
        ]
        test_a = [{"belief": 0.8, "correct": 0.0}, {"belief": 0.1, "correct": 1.0}]
        test_b = [{"belief": 0.8, "correct": 1.0}, {"belief": 0.1, "correct": 0.0}]
        model = public.RidgeLogistic(["margin"]).fit(calibration)
        threshold_a = public.choose_threshold(operating, "belief", 0.4)
        threshold_b = public.choose_threshold(operating, "belief", 0.4)
        self.assertEqual(threshold_a, threshold_b)
        self.assertNotEqual(public.selective(test_a, "belief", threshold_a)["risk"], public.selective(test_b, "belief", threshold_b)["risk"])
        self.assertTrue(np.allclose(model.predict([{"margin": 0.5}]), model.predict([{"margin": 0.5}])))

    def test_incremental_closure_matches_full_recomputation(self):
        stream = synthetic.bench.DynamicKnowledgeStream(seed=4, steps=20)
        full_ms, incremental_ms, _, max_error = synthetic._incremental_runtime(stream, seed=4, repeats=30)
        self.assertGreater(full_ms, 0.0)
        self.assertGreater(incremental_ms, 0.0)
        self.assertEqual(max_error, 0.0)

    def test_prequential_same_timestamp_cannot_see_current_event(self):
        index = chronological.build_index([public.Query(1, 2, 0, (7,))])
        queries = [public.Query(1, 2, 10, (8, 9)), public.Query(1, 2, 11, (8,))]
        rows, _ = chronological.evaluate_prequential(
            queries, index, {q.key: set(q.answers) for q in queries}, 12, 0
        )
        self.assertEqual(rows[0]["winner"], 7)
        self.assertEqual(rows[0]["support"], 1)
        self.assertEqual(rows[1]["winner"], 8)
        self.assertIn((10, 9), index[(1, 2)])

    def test_window_count_guarantee_is_not_minimum_mass(self):
        count, fraction = public.window_deletion_certificate([6, 4, 3], 7)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(fraction, 10 / 13)
        self.assertLess((4 + 3) / 13, fraction)
        for margin in np.linspace(0.1, 13, 27):
            count, _ = public.window_deletion_certificate([6, 4, 3], margin)
            for size in range(count):
                self.assertTrue(all(sum(s) < margin for s in combinations([6, 4, 3], size)))
            self.assertTrue(any(sum(s) >= margin for s in combinations([6, 4, 3], count)))

    def test_full_mass_roundoff_does_not_report_zero_stability(self):
        count, fraction = public.window_deletion_certificate([0.1, 0.2], np.nextafter(0.3, np.inf) + 1e-16)
        self.assertEqual(count, 2)
        self.assertEqual(fraction, 1.0)


if __name__ == "__main__":
    unittest.main()
