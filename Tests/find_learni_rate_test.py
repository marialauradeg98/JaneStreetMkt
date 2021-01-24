import unittest
import find_learn_rate


class TestSmoothLoss(unittest.TestCase):

    def test_filter_loss(self):
        x = (1, 2)
        filtered_x = filter_loss(x, 1/3)
        y = (1, 7/4)

        self.assertAlmostEqual(filtered_x[0], y[0])
        self.assertAlmostEqual(filtered_x[1], y[1])


if __name__ == "__main__":
    unittest.main()
