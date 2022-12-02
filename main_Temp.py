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
                        where a.probeId < 99999999 and (a.measSetNum = 3 or a.measSetNum = 4) and (a.pulseVoltage = 90 or a.pulseVoltage = 93)
                        ORDER BY 1
                        '''
            
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
                        where a.probeId < 99999999 and (a.measSetNum = 3 or a.measSetNum = 4)  
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
        probeType = []
        energy = []

        ## 온도데이터를 계산하기 위하여 새로운 parameter 생성 --> energy 영역 설정.
        for probe_type, volt, cycle, element, prf, pitch, scanrange in zip(AOP_data['probeDescription'], AOP_data['pulseVoltage'], AOP_data['numTxCycles'], 
                                                                           AOP_data['numTxElements'], AOP_data['pulseRepetRate'], AOP_data['probePitchCm'],
                                                                           AOP_data['scanRange']):
            
            if scanrange == 0:
                SR = 0.001
            else:
                SR = scanrange
            energy.append(volt * volt * cycle * element * prf * pitch / SR)

            if "Convex" or "Curved" in probe_type:
                probeType.append(0)
            elif "Linear" in probe_type:
                probeType.append(1)
            else:  ## Phased
                probeType.append(2)

        
        AOP_data['probeType'] = probeType
        AOP_data['energy'] = energy
        AOP_data = AOP_data.fillna(0)
        print(AOP_data.head())

        data = AOP_data[['probeId', 'pulseVoltage', 'numTxCycles', 'numTxElements', 'txFrequencyHz', 'elevAperIndex',
                         'isTxAperModulationEn', 'txpgWaveformStyle', 'scanRange', 'pulseRepetRate', 'probePitchCm',
                         'probeRadiusCm', 'probeElevAperCm0', 'probeElevAperCm1', 'probeNumElements', 'probeElevFocusRangCm',
                         'probeType', 'roomTempC', 'energy']].to_numpy()

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
            model = RandomForestRegressor(n_jobs=-1)
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
            df_import = df_import.append(pd.DataFrame([np.round((model.feature_importances_) * 100, 2)],
                                                      columns=['probeId', 'pulseVoltage', 'numTxCycles', 'numTxElements', 'txFrequencyHz', 'elevAperIndex',
                                                               'isTxAperModulationEn', 'txpgWaveformStyle', 'scanRange', 'pulseRepetRate', 'probePitchCm',
                                                               'probeRadiusCm', 'probeElevAperCm0', 'probeElevAperCm1', 'probeNumElements', 'probeElevFocusRangCm',
                                                               'probeType', 'roomTempC', 'energy']), ignore_index=True)

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

        df = pd.DataFrame(test_input, columns=['probeId', 'pulseVoltage', 'numTxCycles', 'numTxElements',
                                               'txFrequencyHz', 'elevAperIndex', 'isTxAperModulationEn',
                                               'txpgWaveformStyle', 'scanRange', 'pulseRepetRate', 'probePitchCm',
                                               'probeRadiusCm', 'probeElevAperCm0', 'probeElevAperCm1', 'probeNumElements',
                                               'probeElevFocusRangCm', 'probeType', 'roomTempC', 'energy'])

        df_temp = pd.DataFrame(test_target, columns=['temperatureC'])
        df['temperatureC'] = df_temp
        df['temperature_est'] = df_temperature_est
        df.to_csv("temperature_test_input_est.csv")

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
    case = 'model_fit'

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
        df_temperature_est = pd.DataFrame(temperature_est, columns=['temperature_est'])

        df = pd.DataFrame(data, columns=['probeId', 'pulseVoltage', 'numTxCycles', 'numTxElements', 'txFrequencyHz', 'elevAperIndex', 'isTxAperModulationEn',
                                         'txpgWaveformStyle', 'scanRange', 'pulseRepetRate', 'probePitchCm', 'probeRadiusCm', 'probeElevAperCm0', 'probeElevAperCm1',
                                         'probeNumElements', 'probeElevFocusRangCm', 'probeType', 'roomTempC', 'energy'])

        df_temp = pd.DataFrame(target, columns=['temperatureC'])
        
        df['temperatureC'] = df_temp
        df['temperature_est'] = df_temperature_est
        df.to_csv("temperature_est.csv")

        print(df.head())
        mae = mean_absolute_error(target, temperature_est)
        print('|(타깃 - 예측값)|:', mae)