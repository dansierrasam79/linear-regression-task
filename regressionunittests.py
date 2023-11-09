import unittest, pandas as pd 
from main import Data, Database, Computations

class TestLinearRegression(unittest.TestCase):
    def test_readcsvfile(self):
        test_csvfile = Data("dataset\\")
        self.assertIsNone(test_csvfile.readcsvfile(""))
    
    def testcreatevisualization(self):
        test_visualization = Data("dataset\\")
        test_list = pd.read_csv("dataset\\test.csv")
        df_test = pd.DataFrame(test_list, columns=['x','y'])
        self.assertTrue(test_visualization.createvisualization("x", "y", df_test))
    
    def testconnectdb(self):
        test_connectdb = Database("sqlite:///regdataset.db")
        self.assertIsNotNone(test_connectdb.connectdb())

    def testconnectdb2(self):
        test_connectdb2 = Database("")
        self.assertIsNone(test_connectdb2.connectdb())

    def testcreatetable(self):
        test_createtable = Database("sqlite:///regdataset.db")
        self.assertTrue(test_createtable.createtable("test", test_createtable.connectdb()))

    def testcreatetable2(self):
        test_createtable2 = Database("sqlite:///regdataset.db")
        self.assertFalse(test_createtable2.createtable("yelp", test_createtable2.connectdb()))

    def testinsertvalues(self):
        csv_data = Data("dataset\\")
        train = csv_data.readcsvfile("train.csv")
        train_list = []
        for i in range(0,train.last_valid_index()+1):
            train_list.append(list(train.loc[i]))
        test_insertvalues = Database("sqlite:///regdataset.db")
        engine_train = test_insertvalues.connectdb()
        test_insertvalues.createtable("train", engine_train)
        self.assertTrue(test_insertvalues.insertvalues(train_list, "train", engine_train))

    def testreadvalues(self):
        csv_data = Data("dataset\\")
        train = csv_data.readcsvfile("train.csv")
        train_list = []
        for i in range(0,train.last_valid_index()+1):
            train_list.append(list(train.loc[i]))
        test_readvalues = Database("sqlite:///regdataset.db")
        engine_train = test_readvalues.connectdb()
        test_readvalues.createtable("train", engine_train)
        test_readvalues.insertvalues(train_list, "train", engine_train)
        self.assertTrue(test_readvalues.readvalues("train",engine_train))

    def testSSR(self):
        csv_data = Data("dataset\\")
        train = csv_data.readcsvfile("train.csv")
        computeSSR = Computations()
        set_x = 'x' 
        sets_y = ['y1', 'y2', 'y3', 'y4']
        self.assertIsNotNone(computeSSR.SSR(set_x, sets_y, train))
        
if __name__ == "__main__":
    unittest.main()
