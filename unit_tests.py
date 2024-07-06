import unittest
from backend import gpt_scored_rubric_batch  # Replace 'your_module' with the actual module name

class TestGptScoredRubric(unittest.TestCase):

    def test_gpt_scored_rubric(self):
        # Placeholder sentences
        sentences = [
            "Il y a un poulet dans la grotte.",
            "En 1310, le navigateur italien Marco Polo a traversé l'océan Indien.",
            "Cette espèce de plante est très appréciée pour sa beauté."
        ]

        # Expected scores (you need to fill in the expected values)
        expected_scores = [2, 3, 2]

        # Call the function with the test sentences
        scores = gpt_scored_rubric_batch(sentences)

        # Assert that the scores match the expected scores
        self.assertEqual(scores, expected_scores)

    # Add more test cases as needed

if __name__ == '__main__':
    unittest.main()

