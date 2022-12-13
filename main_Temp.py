import tkinter
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import configparser
import numpy as np
import pandas as pd
import pymssql
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.options.mode.chained_assignment = None 



def func_sql_get(server_address, ID, password, list_databases, export_database=None, query=None):
    try:
        if case == 'model_fit':
            sel_database = list_databases
        else:
            ## export_database가 string으로 인식할 수 있기에 list로 변환.
            sel_database = [export_database]
        print(sel_database)

        list_of_df =[]
        for database in sel_database:
            print('-----------------')
            print('connect:', database)
            conn = pymssql.connect(server_address, ID, password, database=database)

            if database == 'New_Trees':
                query = f'''
                        SELECT
                        a.[temperatureId]
                        ,a.[probeId]
                        ,a.[tempSSId]
                        ,a.[measDate]
                        ,a.[measSetNum]
                        ,a.[roomTempC]
                        ,a.[pulseVoltage]
                        ,a.[temperatureC]
                        ,a.[numTxCycles]
                        ,a.[numTxElements]
                        ,a.[txFrequencyHz]
                        ,a.[elevAperIndex]
                        ,a.[isTxAperModulationEn]
                        ,a.[txpgWaveformStyle]
                        ,a.[pulseRepetRate]
                        ,a.[scanRange]
                        ,b.[probeName]
                        ,b.[probePitchCm]
                        ,b.[probeRadiusCm]
                        ,b.[probeElevAperCm0]
                        ,b.[probeElevAperCm1]
                        ,b.[probeNumElements]
                        ,b.[probeElevFocusRangCm] 
                        ,b.[probeDescription]
                        FROM temperature AS a
                        LEFT JOIN probe_geo AS b
                            ON a.[probeId] = b.[probeId]
                        where (a.probeId < 99999999 and a.probeId > 100) and (a.measSetNum = 3 or a.measSetNum = 4) and (a.pulseVoltage = 90 or a.pulseVoltage = 93)
                        ORDER BY 1
                        '''
            ##and (a.pulseVoltage = 90 or a.pulseVoltage = 93)
            else:    
                query = f'''
                        SELECT
                        a.[temperatureId]
                        ,a.[probeId]
                        ,a.[tempSSId]
                        ,a.[measDate]
                        ,a.[measSetNum]
                        ,a.[roomTempC]
                        ,a.[pulseVoltage]
                        ,a.[temperatureC]
                        ,a.[numTxCycles]
                        ,a.[numTxElements]
                        ,a.[txFrequencyHz]
                        ,a.[elevAperIndex]
                        ,a.[isTxAperModulationEn]
                        ,a.[txpgWaveformStyle]
                        ,a.[pulseRepetRate]
                        ,a.[scanRange]
                        ,b.[probeName]
                        ,b.[probePitchCm]
                        ,b.[probeRadiusCm]
                        ,b.[probeElevAperCm0]
                        ,b.[probeElevAperCm1]
                        ,b.[probeNumElements]
                        ,b.[probeElevFocusRangCm] 
                        ,b.[probeDescription]
                        FROM temperature AS a
                        LEFT JOIN probe_geo AS b
                            ON a.[probeId] = b.[probeId]
                        where (a.probeId < 99999999 and a.probeId > 100) and (a.measSetNum = 3 or a.measSetNum = 4)  
                        ORDER BY 1
                        '''

            Raw_data = pd.read_sql(sql=query, con=conn)
            # AOP_data = Raw_data.dropna()
            Raw_data.insert(0, "Database", f'{database}', True)
            ## DataFrame append할 경우, 동일한 parameter 갯수. // ignore_index(True) 인덱스가 기존해의 뒷 번호로 지정
            print('data 갯수:', len(Raw_data.index))
            list_of_df.append(Raw_data)

        AOP_data = pd.concat(list_of_df)
        
        print('-----------------')
        print('Probe 종류 및 갯수')
        print(AOP_data['probeId'].value_counts(dropna=False))

        return AOP_data


    except():
        print('error: func_sql_get')


def func_conf_get():
    try:
        config = configparser.ConfigParser()
        config.read('AOP_config.cfg')

        server_address = config["server address"]["address"]
        databases = config["database"]["name"]
        export_database = config["export database"]["name"]
        ID = config["username"]["ID"]
        password = config["password"]["PW"]

        list_databases = databases.split(',')

        return server_address, ID, password, list_databases, export_database


    except():
        print('error: func_conf_get')


def func_preprocess(AOP_data):
    try:
         ## 누락된 데이터 삭제
        AOP_data = AOP_data.dropna(subset=['probeNumElements'])
        
        probeType = []
        energy = []
        
        ## probeDescription에 데이터가 어떠한 것이 들어있는지 확인.
        print(AOP_data['probeDescription'].unique())
        
        ## 온도데이터를 계산하기 위하여 새로운 parameter 생성 --> energy 영역 설정.
        for probe_type, volt, cycle, element, prf, pitch, scanrange in zip(AOP_data['probeDescription'], AOP_data['pulseVoltage'], AOP_data['numTxCycles'], 
                                                                           AOP_data['numTxElements'], AOP_data['pulseRepetRate'], AOP_data['probePitchCm'],
                                                                           AOP_data['scanRange']):
            
            ## New parameter add: energy(volt, cycle, element, prf, pitch, scanrange)
            if scanrange == 0:
                SR = 0.001
            else:
                SR = scanrange
            ## array 생성(for문 돌려서 각 행마다 변환)
            energy.append(volt * volt * cycle * element * prf * pitch / SR)
            
            
            ## probe_type에 따른 데이터 정렬. 
            if probe_type == 'Linear': 
                Type = 'L'
            elif probe_type == 'Curved':
                Type = 'C'
            elif probe_type == 'Convex':
                Type = 'C'
            elif probe_type == 'Phased':
                Type = 'P'
            elif probe_type == 'AcuNav_Phased':
                Type = 'P'
            ## array 생성(for문 돌려서 각 행마다 변환)
            probeType.append(Type)
        
        ## array 데이터를 데이터프레임 parameter에 입력.
        AOP_data['energy'] = energy        
        AOP_data['probeType'] = probeType
        
        
        ## OneHotEncoder 사용 ==> probeType에 들어있는 데이터를 잘못 계산 혹은 의미있는 데이터로 변환하기 위하여.
        from sklearn.preprocessing import OneHotEncoder
        ohe = OneHotEncoder(sparse=False)
        
        if case == 'model_fit':
            ## fit_transform은 train data에만 사용하고 test data에는 학습된 인코더에 fit만 진행.
            print('model_fit')
            ohe_probe = ohe.fit_transform(AOP_data[['probeType']])
            
        else:
            print('model_predict')
            ohe_proby = ohe.transform(AOP_data[['probeType']])
                    
        ## sklearn.preprocessing.OneHotEncoder를 사용하여 변환된 결과는 numpy.array이기 때문에 이를 데이터프레임으로 변환하는 과정이 필요.
        df_ohe_probe = pd.DataFrame(ohe_probe, columns=['probeType_' + col for col in ohe.categories_[0]])
        
        AOP_data = AOP_data.drop(columns=['probeType'])
        
        
        # df_ohe_probe.reset_index(drop=True, inplace=True)
        
        AOP_data.reset_index(drop=True, inplace=True)
        AOP_data = pd.concat([AOP_data, df_ohe_probe], axis=1)
        
        print(AOP_data.head())
        
        
        AOP_data = AOP_data.fillna(0)
        print(AOP_data.head())

        # data = AOP_data[['probeId', 'pulseVoltage', 'numTxCycles', 'numTxElements', 'txFrequencyHz', 'elevAperIndex',
        #                  'isTxAperModulationEn', 'txpgWaveformStyle', 'scanRange', 'pulseRepetRate', 'probePitchCm',
        #                  'probeRadiusCm', 'probeElevAperCm0', 'probeElevAperCm1', 'probeNumElements', 'probeElevFocusRangCm',
        #                  'probeType_C', 'probeType_L', 'probeType_P','roomTempC', 'energy']].to_numpy()
        
        data = AOP_data[['pulseVoltage', 'numTxCycles', 'numTxElements', 'txFrequencyHz', 'elevAperIndex',
                         'isTxAperModulationEn', 'txpgWaveformStyle', 'scanRange', 'pulseRepetRate', 'probePitchCm',
                         'probeRadiusCm', 'probeElevAperCm0', 'probeElevAperCm1', 'probeNumElements', 'probeElevFocusRangCm',
                         'probeType_C', 'probeType_L', 'probeType_P','roomTempC', 'energy']].to_numpy()
        

        target = AOP_data['temperatureC'].to_numpy()

        return data, target

    except():
        print('error: func_preprocess')


def func_machine_learning(selected_ML, data, target):
    try:
        train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2)

        ## Random Forest 훈련하기.
        if selected_ML == 'RandomForestRegressor':
            from sklearn.ensemble import RandomForestRegressor
            from scipy.stats import uniform, randint
            from sklearn.model_selection import RandomizedSearchCV
            
            ## RandomForest hyperparameter setting.
            # hyperparameter 세팅 시, 진행.
            # n_estimators = randint(20, 100)                             ## number of trees in the random forest
            # max_features = ['auto', 'sqrt']                             ## number of features in consideration at every split
            # max_depth = [int(x) for x in np.linspace(10, 120, num=12)]  ## maximum number of levels allowed in each decision tree
            # min_samples_split = [2, 6, 10]                              ## minimum sample number to split a node
            # min_samples_leaf = [1, 3, 4]                                ## minimum sample number that can be stored in a leaf node
            # bootstrap = [True, False]                                   ## method used to sample data points
            
            # random_grid = {'n_estimators': n_estimators,
            #                 'max_features': max_features,
            #                 'max_depth': max_depth,
            #                 'min_samples_split': min_samples_split,
            #                 'min_samples_leaf': min_samples_leaf,
            #                 'bootstrap': bootstrap}
            
            ## ----------------------
            ## hyperparameter setting
            # param = {'n_estimators': randint(20, 100),                              ## number of trees in the random forest
            #          'min_impurity_decrease': uniform(0.0001, 0.001),               ## number of trees in 
            #          'max_features': ['auto', 'sqrt'],                              ## number of features in consideration at every split
            #          'max_depth': [int(x) for x in np.linspace(10, 120, num=12)],   ## maximum number of levels allowed in each decision tree
            #          'min_samples_split': [2, 6, 10],                               ## minimum sample number to split a node
            #          'min_samples_leaf': [1, 3, 4],                                 ## minimum sample number that can be stored in a leaf node
            #          'bootstrap': [True, False]                                     ## method used to sample data points}
            #          }
            
            
            # # RandomizedSearchCV에서 fit이 완료.
            # rf = RandomForestRegressor()
            # model = RandomizedSearchCV(estimator = rf, param_distributions = param, n_iter = 300, cv = 5, verbose=2, n_jobs = -1)
            
            # model.fit(train_input, train_target)
            # print(model.best_params_)
            ## -----------------------
            
            
            model = RandomForestRegressor(bootstrap='False', max_depth=80, max_features='sqrt', min_impurity_decrease=0.00022562377192675146, min_samples_leaf=1, min_samples_split=2, n_estimators=90, n_jobs=-1)
            scores = cross_validate(model, train_input, train_target, return_train_score=True, n_jobs=-1)
            print()
            print(scores)
            print('Random Forest - Train R^2:', np.round_(np.mean(scores['train_score']), 3))
            print('Random Forest - Train_validation R^2:', np.round_(np.mean(scores['test_score']), 3))

            model.fit(train_input, train_target)
            print('Random Forest - Test R^2:', np.round_(model.score(test_input, test_target), 3))
            prediction = np.round_(model.predict(test_input), 2)

            
            ## each parameter 중요도 출력하여 확인하기.
            df_import = pd.DataFrame()
            # df_import = df_import.append(pd.DataFrame([np.round((model.feature_importances_) * 100, 2)],
            #                                           columns=['probeId', 'pulseVoltage', 'numTxCycles', 'numTxElements', 'txFrequencyHz', 'elevAperIndex',
            #                                                    'isTxAperModulationEn', 'txpgWaveformStyle', 'scanRange', 'pulseRepetRate', 'probePitchCm',
            #                                                    'probeRadiusCm', 'probeElevAperCm0', 'probeElevAperCm1', 'probeNumElements', 'probeElevFocusRangCm',
            #                                                    'probeType_C', 'probeType_L','probeType_P', 'roomTempC', 'energy']), ignore_index=True)
            df_import = df_import.append(pd.DataFrame([np.round((model.feature_importances_) * 100, 2)],
                                                      columns=['pulseVoltage', 'numTxCycles', 'numTxElements', 'txFrequencyHz', 'elevAperIndex',
                                                               'isTxAperModulationEn', 'txpgWaveformStyle', 'scanRange', 'pulseRepetRate', 'probePitchCm',
                                                               'probeRadiusCm', 'probeElevAperCm0', 'probeElevAperCm1', 'probeNumElements', 'probeElevFocusRangCm',
                                                               'probeType_C', 'probeType_L','probeType_P', 'roomTempC', 'energy']), ignore_index=True)

            mae = mean_absolute_error(test_target, prediction)
            print('|(타깃 - 예측값)|:', mae)
            print(df_import)

        ## modeling file 저장 장소.
        newpath = './Model'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        joblib.dump(model, f'Model/{selected_ML}_v1_python37.pkl')

        loaded_model = joblib.load('Model/RandomForestRegressor_v1_python37.pkl')
        temperature_est = loaded_model.predict(test_input)
        df_temperature_est = pd.DataFrame(temperature_est, columns=['temperature_est'])

        # df = pd.DataFrame(test_input, columns=['probeId', 'pulseVoltage', 'numTxCycles', 'numTxElements',
        #                                        'txFrequencyHz', 'elevAperIndex', 'isTxAperModulationEn',
        #                                        'txpgWaveformStyle', 'scanRange', 'pulseRepetRate', 'probePitchCm',
        #                                        'probeRadiusCm', 'probeElevAperCm0', 'probeElevAperCm1', 'probeNumElements',
        #                                        'probeElevFocusRangCm', 'probeType_C', 'probeType_L','probeType_P', 'roomTempC', 'energy'])
        
        df = pd.DataFrame(test_input, columns=['pulseVoltage', 'numTxCycles', 'numTxElements',
                                               'txFrequencyHz', 'elevAperIndex', 'isTxAperModulationEn',
                                               'txpgWaveformStyle', 'scanRange', 'pulseRepetRate', 'probePitchCm',
                                               'probeRadiusCm', 'probeElevAperCm0', 'probeElevAperCm1', 'probeNumElements',
                                               'probeElevFocusRangCm', 'probeType_C', 'probeType_L','probeType_P', 'roomTempC', 'energy'])

        df_temp = pd.DataFrame(test_target, columns=['temperatureC'])
        df['temperatureC'] = df_temp
        df['temperature_est'] = df_temperature_est
        df.to_csv("temperature_train_score.csv")

        if selected_ML == 'RandomForestRegressor':
            pass
            # func_feature_import()

        else:
            pass

    except():
        print('error: func_machine_learning')


## main
## 데이터 몇개가 들어가야 성능이 향상되는 지 확인.
## 1) 일반적으로 데이터 갯수를 주파수에 따라서 한개씩만 넣어서 확인.
if __name__ == '__main__':
    case = 'model_predict'

    if case == 'model_fit':
        server_address, ID, password, list_databases, export_database = func_conf_get()
        AOP_data = func_sql_get(server_address=server_address, ID=ID, password=password, list_databases=list_databases)
        print('-----------------')
        print('데이터 총합:', len(AOP_data.index))
        data, target = func_preprocess(AOP_data=AOP_data)
        ## RandomForestRegressor modeling
        func_machine_learning(selected_ML='RandomForestRegressor', data=data, target=target)


    if case == 'model_predict':
        server_address, ID, password, list_databases, export_database = func_conf_get()
        AOP_data = func_sql_get(server_address=server_address, ID=ID, password=password, list_databases=list_databases,
                                export_database=export_database)

        data, target = func_preprocess(AOP_data=AOP_data)

        loaded_model = joblib.load('Model/RandomForestRegressor_v1_python37.pkl')
        temperature_est = loaded_model.predict(data)
        print('model_predict', temperature_est)
        df_temperature_est = pd.DataFrame(temperature_est, columns=['temperature_est'])

        df = pd.DataFrame(data, columns=['probeId', 'pulseVoltage', 'numTxCycles', 'numTxElements', 'txFrequencyHz', 'elevAperIndex', 'isTxAperModulationEn',
                                         'txpgWaveformStyle', 'scanRange', 'pulseRepetRate', 'probePitchCm', 'probeRadiusCm', 'probeElevAperCm0', 'probeElevAperCm1',
                                         'probeNumElements', 'probeElevFocusRangCm', 'probeType_C', 'roomTempC', 'energy'])

        df_temp = pd.DataFrame(target, columns=['temperatureC'])
        
        df['temperatureC'] = df_temp
        df['temperature_est'] = df_temperature_est
        df.to_csv("temperature_est.csv")

        print(df.head())
        mae = mean_absolute_error(target, temperature_est)
        print('|(타깃 - 예측값)|:', mae)