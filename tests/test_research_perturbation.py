import unittest
from mmresearch.research_perturbation import compare_adoption


class PerturbationUptakeTests(unittest.TestCase):
    def test_only_new_exposed_terms_are_counted(self):
        normal = [{"id": "a", "prediction": "A red dog.", "context": [{"caption": "A red dog."}]}]
        changed = [{"id": "a", "prediction": "A blue dog.", "context": [{"caption": "A blue dog.",
                    "perturbations": [{"from": "red", "to": "blue"}, {"from": "one", "to": "two"}]}]}]
        result = compare_adoption(normal, changed)
        self.assertEqual(result["details"][0]["inserted_terms"], ["blue"])
        self.assertEqual(result["newly_adopts"], 1)
        self.assertEqual(result["paired_effect"]["delta"], 1.)

    def test_preexisting_context_terms_are_not_new_exposure(self):
        normal = [{"id": "a", "prediction": "A dog.", "context": [{"caption": "A red and blue ball."}]}]
        changed = [{"id": "a", "prediction": "A blue ball.", "context": [{"caption": "A blue ball.",
                    "perturbations": [{"from": "red", "to": "blue"}]}]}]
        self.assertEqual(compare_adoption(normal, changed)["status"], "no_eligible_interventions")

    def test_ids_cannot_be_silently_intersected(self):
        with self.assertRaises(ValueError):
            compare_adoption([{"id": "a"}], [{"id": "b"}])
