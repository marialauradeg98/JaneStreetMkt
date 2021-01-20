import unittest
import data_visualization_main
import numpy as np
import pandas as pd

class TestDataVisualization(unittest.TestCase):
    """
    Unit tests for data visualization module
    """
    def test_daily_avarage(self):
        #create a test dataset
        data = {
            "date":[0,0,0,1,1,1],
            "Value1":[2,6,19,1,12,11],
            "Value2":[4,5,9,1,11,0]}
        df = pd.DataFrame(data)
        #compute results with the function we want to test
        res=data_visualization_main.daily_avarage(df)
        #put the results in np.array
        res=np.ravel(res)
        res=np.array(res)
        #built an array with right valuea for mean Value1 and Value2
        true=np.array([9,6,8,4])
        #verify the elements are equal each other for the two np.array
        for i in range(4):
            self.assertEqual(res[i],true[i])

    def test_compute_profit(self):
        #create a test dataset
        data = {
            "date":[0,0,1,1,2,2],
            "weighted_resp":[1,2,3,4,5,6],
            "action":[1,2,3,4,5,6]}
        df=pd.DataFrame(data)
        #comupute profit and day of trading using the function we want to test
        res_u,res_day=data_visualization_main.compute_profit(df)
        #verify if the fuction returns 2 trading days
        self.assertEqual(res_day,2)
        #verify if the function returns the right profit
        self.assertEqual(res_u,546)

if __name__ == "__main__":
    unittest.main()
