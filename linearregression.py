import pandas as pd
import numpy as np
import csv
import sqlalchemy as db
import os, math
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

class Data:
    def __init__(self, path):
        self.path = path

    def readcsvfile(self, dataset):
        '''
        Reads csv file if file is available and returns it as a dataframe 
        '''
        if os.path.isfile(self.path+dataset):
            final_set = pd.read_csv(self.path+dataset)
            return final_set
        else:
            return None
    
    def createvisualization(self,x,y, title, dataset):
        '''
        Creates and returns the visualization for the test and selected ideal function data
        '''
        if len(dataset) > 0:
            # displays the title, x & y labels and zooms each visualization
            plt.title(title)
            plt.xlabel(x)
            plt.ylabel(y)
            # draw regplot using Seaborn without confidence interval
            rval = sns.regplot(x = x,y = y,data = dataset, ci=None)
            rval.figure.set_size_inches(10,6)
            # show the plot
            plt.show()
            return True
        else:
            return False

class Database(Data):
    def __init__(self, path):
        self.path = path
        Data.__init__(self, path)
    
    def connectdb(self):
        '''
        Connects to the sqlite database using sqlalchemy and returns an engine object
        '''
        try:
            # used sqlalchemy 1.4.49 version (maintenance)
            engine = db.create_engine(self.path, echo = True)
            return engine
        except:
            return None

    def createtable(self, table, engine):
        '''
        Creates a table for the train, ideal and test sets and returns true
        '''
        # get meta data object
        meta_data = db.MetaData()
        # get connection object
        connection = engine.connect()
        table_list = ["train", "ideal", "test"]
        if table in table_list:
            if table == "train":
                train_table = db.Table("train", meta_data,
                db.Column("x", db.Float, nullable=False),
                db.Column("y1", db.Float, nullable=False),
                db.Column("y2", db.Float, nullable=False),
                db.Column("y3", db.Float, nullable=False),
                db.Column("y4", db.Float, nullable=False))

            elif table == "ideal":
                ideal_table = db.Table("ideal", meta_data,
                db.Column("x", db.Float, nullable=False),db.Column("y1", db.Float, nullable=False),db.Column("y2", db.Float, nullable=False),db.Column("y3", db.Float, nullable=False),
                db.Column("y4", db.Float, nullable=False), db.Column("y5", db.Float, nullable=False), db.Column("y6", db.Float, nullable=False), db.Column("y7", db.Float, nullable=False),
                db.Column("y8", db.Float, nullable=False),db.Column("y9", db.Float, nullable=False),db.Column("y10", db.Float, nullable=False),db.Column("y11", db.Float, nullable=False),
                db.Column("y12", db.Float, nullable=False),db.Column("y13", db.Float, nullable=False),db.Column("y14", db.Float, nullable=False),db.Column("y15", db.Float, nullable=False),
                db.Column("y16", db.Float, nullable=False),db.Column("y17", db.Float, nullable=False),db.Column("y18", db.Float, nullable=False),db.Column("y19", db.Float, nullable=False),
                db.Column("y20", db.Float, nullable=False),db.Column("y21", db.Float, nullable=False),db.Column("y22", db.Float, nullable=False),db.Column("y23", db.Float, nullable=False),
                db.Column("y24", db.Float, nullable=False),db.Column("y25", db.Float, nullable=False),db.Column("y26", db.Float, nullable=False),db.Column("y27", db.Float, nullable=False),
                db.Column("y28", db.Float, nullable=False),db.Column("y29", db.Float, nullable=False),db.Column("y30", db.Float, nullable=False),db.Column("y31", db.Float, nullable=False),
                db.Column("y32", db.Float, nullable=False),db.Column("y33", db.Float, nullable=False),db.Column("y34", db.Float, nullable=False),db.Column("y35", db.Float, nullable=False),
                db.Column("y36", db.Float, nullable=False),db.Column("y37", db.Float, nullable=False),db.Column("y38", db.Float, nullable=False),db.Column("y39", db.Float, nullable=False),
                db.Column("y40", db.Float, nullable=False),db.Column("y41", db.Float, nullable=False),db.Column("y42", db.Float, nullable=False),db.Column("y43", db.Float, nullable=False),
                db.Column("y44", db.Float, nullable=False),db.Column("y45", db.Float, nullable=False),db.Column("y46", db.Float, nullable=False),db.Column("y47", db.Float, nullable=False),
                db.Column("y48", db.Float, nullable=False),db.Column("y49", db.Float, nullable=False),db.Column("y50", db.Float, nullable=False))

            elif table == "test":
                test_table = db.Table("test", meta_data,
                db.Column("x", db.Float, nullable=False),
                db.Column("y", db.Float, nullable=False),
                db.Column("ydev", db.Float, nullable=False),
                db.Column("yideal", db.Float, nullable=False))
            
            # create test table and store the information in metadata
            meta_data.create_all(engine)
            return True
        else: 
            return False

    def insertvalues(self, dataset, table, engine):
        '''
        Inserts values into the created table for train, ideal and test and returns True
        '''
        table_list = ["train", "ideal", "test"]
        if table in table_list:
            if table == "train":
                # get meta data object
                meta_data = db.MetaData()
        
                # get connection object
                connection = engine.connect()
        
                # set actor creation script table
                train_table = db.Table(table, meta_data,autoload = True, autoload_with=engine)
    
                # insert data
                for lst in dataset:
                    data = train_table.insert().values(x=lst[0], y1=lst[1], y2=lst[2],y3=lst[3],y4=lst[4])
                    # execute the insert statement
                    connection.execute(data)
                return True
            
            elif table == "ideal":
                # get meta data object
                meta_data = db.MetaData()
        
                # get connection object
                connection = engine.connect()
        
                # set actor creation script table
                ideal_table = db.Table(table, meta_data,autoload = True, autoload_with=engine)

                # insert data
                for lst in dataset:
                    data = ideal_table.insert().values(x=lst[0], y1=lst[1], y2=lst[2],y3=lst[3],y4=lst[4], y5=lst[5], 
                                                 y6=lst[6],y7=lst[7],y8=lst[8], y9=lst[9], y10=lst[10], y11=lst[11], 
                                                 y12=lst[12],y13=lst[13],y14=lst[14], y15=lst[15], y16=lst[16],y17=lst[17],
                                                 y18=lst[18], y19=lst[19], y20=lst[20], y21=lst[21], y22=lst[22],y23=lst[23],
                                                 y24=lst[24], y25=lst[25], y26=lst[26],y27=lst[27],y28=lst[28], y29=lst[29], 
                                                 y30=lst[30], y31=lst[31], y32=lst[32],y33=lst[33],y34=lst[34], y35=lst[35], 
                                                 y36=lst[36],y37=lst[37],y38=lst[38], y39=lst[39], y40=lst[40], 
                                                 y41=lst[41], y42=lst[42],y43=lst[43],y44=lst[44], y45=lst[45], 
                                                 y46=lst[46],y47=lst[47],y48=lst[48], y49=lst[49],y50=lst[50])
                    # execute the insert statement
                    connection.execute(data)
                return True
 
            elif table == "test":
                # get meta data object
                meta_data = db.MetaData()
        
                # get connection object
                connection = engine.connect()
        
                # set actor creation script table
                test_table = db.Table(table, meta_data,autoload = True, autoload_with=engine)
    
                # insert data
                for lst in dataset:
                    data = test_table.insert().values(x=lst[0], y=lst[1], ydev=lst[2],yideal=lst[3])
                    # execute the insert statement
                    connection.execute(data)
                return True
        
        else:
            return False
            
    def readvalues(self, table, engine):
        '''
        Reads values from the created tables for train, ideal and test and prints selected values
        '''
        table_list = ["train", "ideal", "test"]
        if table in table_list:
            if table == "train":
                # get meta data object
                meta_data = db.MetaData()
                
                # get connection object
                connection = engine.connect()
    
                # set actor creation script table
                train_table = db.Table("train", meta_data,autoload=True, autoload_with=engine)

                # set the select statement
                select_train = train_table.select()
    
                # execute the select statement
                result = connection.execute(select_train)
                
                alltrainrows = result.fetchmany(5)

                for row in alltrainrows:
                    print(row)
                
                return True
            
            elif table == "ideal":
                # get meta data object
                meta_data = db.MetaData()
                
                # get connection object
                connection = engine.connect()
    
                # set actor creation script table
                ideal_table = db.Table("ideal", meta_data,autoload=True, autoload_with=engine)

                # set the select statement
                select_ideal = ideal_table.select()
    
                # execute the select statement
                result = connection.execute(select_ideal)
                
                allidealrows = result.fetchmany(5)

                for row in allidealrows:
                    print(row)
                
                return True
            
            elif table == "test":
                # get meta data object
                meta_data = db.MetaData()
                
                # get connection object
                connection = engine.connect()
    
                # set actor creation script table
                test_table = db.Table("test", meta_data,autoload=True, autoload_with=engine)

                # set the select statement
                select_test = test_table.select()
    
                # execute the select statement
                result = connection.execute(select_test)
                
                alltestrows = result.fetchmany(5)

                for row in alltestrows:
                    print(row)
                return True
        else:
            return False

class Computations:
    def SSR(self,set_x, sets_y, dataset):
        '''
        Computes SSR for x-y value pairs and saves the deviations for later use in the project
        '''
        if len(dataset) > 0:
            SSR_list, deviations_list,SSR_deviations_list  = [], [], []
        
            for y_value in sets_y:
                x = dataset[list(set_x)]
                y = dataset[y_value]
                regression_model = LinearRegression()
                regression_model.fit(x, y)
                y_pred = regression_model.predict(x)
                df_dev = pd.DataFrame({'Actual':y, 'Predicted':y_pred})
                SSR_list.append(str(np.sum(np.square(df_dev['Predicted']-df_dev['Actual']))))
                deviations_list = list(df_dev['Predicted']-df_dev['Actual'])
                SSR_deviations_list.append(SSR_list)
                SSR_deviations_list.append(deviations_list)
            return SSR_deviations_list
        else:
            return None

if __name__ == "__main__":
    # Read csv files
    # Due to inheriting the methods from the Data class, Database objects can also use the readcsvfile & createvisualization methods
    read_visualize_data = Database("dataset\\")
    
    train = read_visualize_data.readcsvfile("train.csv")
    ideal = read_visualize_data.readcsvfile("ideal.csv")
    test = read_visualize_data.readcsvfile("test.csv")

    if len(train) == 0 or len(ideal) == 0 or len(test) == 0:
        raise Exception("Error. Incorrect path or CSV filename.")
    
    train_list, ideal_list, test_list = [], [], []
    
    for i in range(0,train.last_valid_index()+1):
        train_list.append(list(train.loc[i]))

    for j in range(0,ideal.last_valid_index()+1):
        ideal_list.append(list(ideal.loc[j]))
    
    for k in range(0,test.last_valid_index()+1):
        test_list.append(list(test.loc[k]))

    # Create table for train and ideal
    tables = Database("sqlite:///linearregressioncomp.db")
    engine_train = tables.connectdb()
    if engine_train != None:
        tables.createtable("train",engine_train)
        # insert values
        tables.insertvalues(train_list, "train", engine_train)
    else:
        raise Exception("Engine object was not created")
    
    engine_ideal = tables.connectdb()
    if engine_ideal != None:
        tables.createtable("ideal",engine_ideal)
        # insert values
        tables.insertvalues(ideal_list, "ideal", engine_ideal)
    else:
        raise Exception("Engine object was not created")
    
    # Compute SSR for train and ideal
    computeSSR = Computations()
    train_set_x = 'x' 
    train_sets_y = ['y1', 'y2', 'y3', 'y4']
    train_SSR = computeSSR.SSR(train_set_x, train_sets_y, train)
    train_SSR_dict = {}
    for i in range(0,len(train_SSR[0])):
        train_SSR_dict['y' + str(i+1)] = train_SSR[0][i]
                   
    ideal_set_x = 'x' 
    ideal_sets_y = ['y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10',
             'y11', 'y12', 'y13', 'y14', 'y15', 'y16', 'y17', 'y18', 'y19', 'y20',
             'y21', 'y22', 'y23', 'y24', 'y25', 'y26', 'y27', 'y28', 'y29', 'y30',
             'y31', 'y32', 'y33', 'y34', 'y35', 'y36', 'y37', 'y38', 'y39', 'y40',
             'y41', 'y42', 'y43', 'y44', 'y45', 'y46', 'y47', 'y48', 'y49', 'y50']
    
    ideal_SSR = computeSSR.SSR(ideal_set_x, ideal_sets_y, ideal)
    
    # Determine which ideal functions map closest to train
    min_ssr_diff_list, min_ssr_diff_final = [],[]
    for train_ssr_val in train_SSR[0]:
        min_ssr_diff_list = []
        for ideal_ssr_val in ideal_SSR[0]:
            min_ssr_diff_list.append(abs(float(train_ssr_val)-float(ideal_ssr_val)))
        min_ssr_diff_final.append(min_ssr_diff_list.index(min(min_ssr_diff_list))+1)
    
    min_ssr_values = []
    for pos in min_ssr_diff_final:
        min_ssr_values.append(ideal_SSR[0][pos-1])
    
    min_ssr_final = {}
    for i in range(0, len(min_ssr_values)):
        min_ssr_final['y' + str(min_ssr_diff_final[i])] = min_ssr_values[i]

    # Compute test SSR
    test_set_x = 'x' 
    test_sets_y = ['y']
    test_SSR_dict = {}
    test_SSR = computeSSR.SSR(test_set_x, test_sets_y, test)
    test_SSR_dict['y' + str(1)] = test_SSR[0][0]

    # Determine which ideal function SSR value does not exceed test SSR by sqrt(2)
    for value in test_SSR[0]:
        for value2 in min_ssr_values:
            upper_bound = float(value)*math.sqrt(2)
            lower_bound = float(value)/math.sqrt(2)
            if float(value2) < upper_bound and float(value2) > lower_bound:
                matched_ideal_fn = min_ssr_diff_final[min_ssr_values.index(value2)]
    
    # Find out how many values in test map to each of the functions and select that ideal func
    test_list_y = []
    for lst in test_list:
        test_list_y.append(lst[1])
    
    ideal_func_list = []
    for lst in ideal_list:
        ideal_func = []
        ideal_func.append(lst[min_ssr_diff_final[0]]) 
        ideal_func.append(lst[min_ssr_diff_final[1]]) 
        ideal_func.append(lst[min_ssr_diff_final[2]]) 
        ideal_func.append(lst[min_ssr_diff_final[3]])
        ideal_func_list.append(ideal_func)

    ideal_fn, ideal_fn2, ideal_fn3, ideal_fn4, ideal_fn_final = [], [], [], [], []
    for y in test_list_y:
        for lst in ideal_func_list:
            if y > (lst[0]/math.sqrt(2)) and y < (lst[0]*math.sqrt(2)):
                ideal_fn.append(y)
            elif y > (lst[1]/math.sqrt(2)) and y < (lst[1]*math.sqrt(2)):
                ideal_fn2.append(y)
            elif y > (lst[2]/math.sqrt(2)) and y < (lst[2]*math.sqrt(2)):
                ideal_fn3.append(y)
            elif y > (lst[3]/math.sqrt(2)) and y < (lst[3]*math.sqrt(2)):
                ideal_fn4.append(y)
    
    ideal_fn_final_dict = {}
    ideal_fn_final.append(len(set(ideal_fn)))
    ideal_fn_final.append(len(set(ideal_fn2)))
    ideal_fn_final.append(len(set(ideal_fn3)))
    ideal_fn_final.append(len(set(ideal_fn4)))

    for i in range(0, len(ideal_fn_final)):
        ideal_fn_final_dict[min_ssr_diff_final[i]] = ideal_fn_final[i]
    
    ideal_max_value = max(ideal_fn_final_dict.values())

    for k,v in ideal_fn_final_dict.items():
        if v == ideal_max_value:
            ideal_max_key = "y" + str(k)
    
    # Add test x-y values, test deviation and ideal func y-values into db
    test_x_final_list, test_y_final_list, test_dev_final_list, ideal_func_final_list = [],[],[],[]
    for i in range(0, len(test_list)):
        test_x_final_list.append(test_list[i][0])
        test_y_final_list.append(test_list[i][1])
        test_dev_final_list.append(test_SSR[1][i])
    
    for j in range(0, len(ideal_func_list)):
        ideal_func_final_list.append(ideal_func_list[j][1])
    
    for k in range(0,300):
        test_x_final_list.append(0)
        test_y_final_list.append(0)
        test_dev_final_list.append(0)
    
    test_final = [] 
    for pos_val in range(0, len(ideal_func_final_list)):
        test_db_list = []
        test_db_list.append(test_x_final_list[pos_val])
        test_db_list.append(test_y_final_list[pos_val])
        test_db_list.append(test_dev_final_list[pos_val])
        test_db_list.append(ideal_func_final_list[pos_val])
        test_final.append(test_db_list)
    
    engine_test = tables.connectdb()
    if engine_test != None:
        tables.createtable("test",engine_test)
        # insert values
        tables.insertvalues(test_final, "test", engine_test)
    else:
        raise Exception("Engine object was not created")

    # Read 5 values each from train, ideal and test db tables
    print('Reading values from database...')
    tables.readvalues("train", engine_train)
    tables.readvalues("ideal", engine_ideal)
    tables.readvalues("test", engine_test)
    
    #Output
    print()
    print("Final Output Summary:")
    print("Train SSR", train_SSR_dict)
    print("Closest ideal functions after comparison with train SSR values: ", min_ssr_final)
    print("Test SSR", test_SSR_dict)
    print("Matched Ideal Function to Test SSR: ", 'y' + str(matched_ideal_fn))
    print(ideal_max_key + " is the matching ideal function based on", ideal_max_value ,"out of 100 'close' values")
    print()

    # Add matplotlib visualization for test and selected ideal function
    df_test = pd.DataFrame(test_list, columns=['x','y'])
    read_visualize_data.createvisualization('x', 'y', 'test', df_test)
    # create df for ideal function visualization
    ideal_x = ideal['x'].to_list()
    ideal_y = ideal[ideal_max_key].to_list()
    ideal_final_list = []
    for i in range(0, len(ideal_x)):
        ideal_list_final = []
        ideal_list_final.append(ideal_x[i])
        ideal_list_final.append(ideal_y[i])
        ideal_final_list.append(ideal_list_final)
    df_ideal = pd.DataFrame(ideal_final_list, columns = ['x', ideal_max_key])
    read_visualize_data.createvisualization('x', ideal_max_key, 'Selected Ideal Function', df_ideal)

    # Please check the regressionunittests.py file for unit tests 