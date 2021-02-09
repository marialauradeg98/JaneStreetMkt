import unittest
import JaneStreetMkt.splitting as splitting
import numpy as np
import pandas as pd

class TestSplitting(unittest.TestCase):
    """
    Unit tests for splitting module
    """
    def test_data_split_one(self):
        #create a test dataset
        data = {
            "date":[10,20,30,40,50,60,70,80,90,100],
            "action":[1,2,3,4,5,6,7,8,9,10]}
        df = pd.DataFrame(data)
        #compute results with the function we want to test
        X_tr,y_tr,X_ts,y_ts=splitting.split_data(df)
        #put the results in np.array
        X_train=np.ravel(X_tr)
        y_train=np.array(y_tr)
        X_test=np.ravel(X_ts)
        y_test=np.ravel(y_ts)
        #built arrays with right values for mean Value1 and Value2
        true_xtr=np.array([10,20,30,40,50,60,70,80])
        true_xts=np.array([90,100])
        true_ytr=np.array([1,2,3,4,5,6,7,8])
        true_yts=np.array([9,10])
        #verify the elements are equal each other for the two np.array
        for i in range(8):
            self.assertEqual(X_train[i],true_xtr[i])
            self.assertEqual(y_train[i],true_ytr[i])
        for i in range(2):
            self.assertEqual(X_test[i],true_xts[i])
            self.assertEqual(y_test[i],true_yts[i])
    def test_data_split_two(self):
        #create a test dataset
        data = {
            "date":[10,20,30,40,50,60,70,80,90,100],
            "action":[1,2,3,4,5,6,7,8,9,10]}
        df = pd.DataFrame(data)
        #compute results with the function we want to test
        X_tr,y_tr,X_ts,y_ts,X_vl,y_vl=splitting.split_data(df,val=True)
        #put the results in np.array
        X_train=np.ravel(X_tr)
        y_train=np.array(y_tr)
        X_test=np.ravel(X_ts)
        y_test=np.ravel(y_ts)
        X_val=np.ravel(X_vl)
        y_val=np.ravel(y_vl)
        print(X_train,y_train,X_test,y_test,X_val,y_val)
        #built arrays with right values for mean Value1 and Value2
        true_xtr=np.array([10,20,30,40,50,60])
        true_xts=np.array([90,100])
        true_ytr=np.array([1,2,3,4,5,6])
        true_yts=np.array([9,10])
        true_yvl=np.array([7,8])
        true_xvl=np.array([70,80])
        #verify the elements are equal each other for the two np.array
        for i in range(6):
            self.assertEqual(X_train[i],true_xtr[i])
            self.assertEqual(y_train[i],true_ytr[i])
        for i in range(2):
            self.assertEqual(X_test[i],true_xts[i])
            self.assertEqual(y_test[i],true_yts[i])
        for i in range(2):
            self.assertEqual(X_val[i],true_xvl[i])
            self.assertEqual(y_val[i],true_yvl[i])

if __name__ == "__main__":
    unittest.main()
