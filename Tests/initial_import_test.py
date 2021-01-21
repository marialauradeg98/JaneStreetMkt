import unittest
import initial_import
import numpy as np
import numpy.testing as nt


class TestImport(unittest.TestCase):

    def test_import(self):
        # import first ten columns
        data = initial_import.import_dataset(10)
        ts_id = data["ts_id"].values.tolist()
        x = [i for i in range(10)]
        self.assertListEqual(ts_id, x)

    def test_fast_import(self):
        # import first ten columns
        data = initial_import.import_dataset_faster(10)
        ts_id = data["ts_id"].values.tolist()
        x = [i for i in range(10)]
        self.assertListEqual(ts_id, x)

    def test_sampl_import(self):
        # import first ten columns
        data = initial_import.import_sampled_dataset(10, 10)
        ts_id = data["ts_id"].values.tolist()
        x = [i for i in range(9, 109, 10)]
        self.assertListEqual(ts_id, x)

    def test_import_training(self):
        # import first ten columns
        data = initial_import.import_training_set(rows=10)
        ts_id = data["ts_id"].values.tolist()
        x = [i for i in range(10)]
        self.assertListEqual(ts_id, x)

    def test_compute_action(self):
        x = [1, 0, 1, 0, 0, 1, 1, 1, 0, 1]
        data = initial_import.import_dataset(10)
        action = data["action"].values.tolist()
        self.assertListEqual(action, x)


if __name__ == "__main__":
    unittest.main()
