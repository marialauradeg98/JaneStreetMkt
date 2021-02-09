import unittest
import JaneStreetMkt.feature_selection as feature_selection
import numpy as np
import pandas as pd


class TestImport(unittest.TestCase):

    def test_remove_feat(self):
        # create dataframe
        data = {
            "Name": ["Paolo", "Marialaura"],
            "Surname": ["Spina", "De Grazia"]}
        df = pd.DataFrame(data)
        remove = ["Spina"]
        # remove spina from dataframe
        new_data = feature_selection.remove_features(df, remove)
        student = new_data.iloc[0, 1]
        # if spina is removed the The first student should be de grazia
        self.assertEqual(student, "De Grazia")

    def test_remove_redundant_feat(self):
        # create dataframe chess tournament
        data = {
            "Student 1": ["De Grazia", "De Grazia", "Carlsen"],
            "Student 2": ["Spina", "Carlsen", "Spina"],
            "Chess Games": [12, 10, 5]}
        col_order = ["Student 1", "Student 2", "Chess Games"]
        df = pd.DataFrame(data)
        df = df[col_order]

        # series counting total number of wins in the tournament
        x = pd.Series([13, 14, 0],
                      index=["De Grazia", "Carlsen", "Spina"])

        # 2 pairings with more than 6 Games
        # Spina - De Grazia: Spina is deleted since it has lower number of wins
        # De Grazia - Carlsen: De Grazia eliminated since it has lower number of wins
        new_data, worst_players = feature_selection.remove_redundat_feat(df, x, 6)
        self.assertEqual(worst_players, ["Spina", "De Grazia"])

    def test_remove_duplicates(self):
        data = {
            "Name": ["Marialaura", "Marialaura", "Paolo"],
            "Surname": ["De Grazia", "Rossi", "Spina"],
            "Mark": [27, 27, 18]}
        col_order = ["Name", "Surname", "Mark"]
        df = pd.DataFrame(data)
        df = df[col_order]
        # Remove Marialaura from the dataframe
        # Since the surname Rossi is different Rossi wont be inserted in the list
        # of redundant features but it will be deleted from the Dataframe
        # because every  row containing Marialaura will be deleted
        no_dupl, feat = feature_selection.remove_duplicates(df, 21)
        # if everything is ok Spina should be ethe first elemetof the dataframe
        student = no_dupl.iloc[0, 1]
        self.assertEqual(student, "Spina")
        # if everything ok only Marialaura should be eliminated
        self.assertEqual(feat, ["Marialaura"])


if __name__ == "__main__":
    unittest.main()
