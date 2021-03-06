import tkinter
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import configparser
import numpy as np
import pandas as pd
import pymssql

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def func_sql_get(server_address, ID, password, list_databases, query=None):
    try:
        for database in list_databases:

            print(database)
            conn = pymssql.connect(server_address, ID, password, database=database)

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
                    --where a.probeId < 99999999 and (a.measSetNum = 4)
                    ORDER BY 1
                    '''

            Raw_data = pd.read_sql(sql=query, con=conn)
            # AOP_data = Raw_data.dropna()
            Raw_data.insert(0, "Database", f'{database}', True)
            AOP_data = Raw_data.append(Raw_data, ignore_index=True)                                                     ## DataFrame append??? ??????, ????????? parameter ??????. // ignore_index(True) ???????????? ???????????? ??? ????????? ??????

        print(Raw_data['probeId'].value_counts(dropna=False))

        return AOP_data


    except():
        print('error: func_sql_get')


def func_conf_get():
    try:
        config = configparser.ConfigParser()
        config.read('AOP_config.cfg')

        server_address = config["server address"]["address"]
        databases = config["database"]["name"]
        ID = config["username"]["ID"]
        password = config["password"]["PW"]

        list_databases = databases.split(',')

        return server_address, ID, password, list_databases


    except():
        print('error: func_conf_get')


def func_preprocess(AOP_data):
    try:
        Nor_temp = []
        probeType = []

        for temp, amb, prf, probe_type in zip(AOP_data['temperatureC'], AOP_data['roomTempC'], AOP_data['pulseRepetRate'], AOP_data['probeDescription']):
            Nor_temp.append((temp - amb) / prf)

            if "Convex" or "Curved" in probe_type:
                probeType.append(0)
            elif "Linear" in probe_type:
                probeType.append(1)
            else:                       ## Phased
                probeType.append(2)

        AOP_data['probeType'] = probeType
        AOP_data['Nor_Temp'] = Nor_temp
        AOP_data = AOP_data.fillna(0)

        print(AOP_data.head())

        data = AOP_data[['probeId', 'pulseVoltage', 'numTxCycles', 'numTxElements', 'txFrequencyHz', 'elevAperIndex',
                         'isTxAperModulationEn', 'txpgWaveformStyle', 'scanRange', 'probePitchCm', 'probeRadiusCm',
                         'probeElevAperCm0', 'probeElevAperCm1', 'probeNumElements', 'probeElevFocusRangCm',
                         'probeType']].to_numpy()

        target = AOP_data['Nor_Temp'].to_numpy()

        return data, target

    except():
        print('error: func_preprocess')

def func_machine_learning(selected_ML, data, target):
    try:
        train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2)

        ## Random Forest ????????????.
        if selected_ML == 'RandomForestRegressor':
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_jobs=-1)
            scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)
            print()
            print(scores)
            print('Random Forest - Train R^2:', np.round_(np.mean(scores['train_score']), 3))
            print('Random Forest - Train_validation R^2:', np.round_(np.mean(scores['test_score']), 3))

            rf.fit(train_input, train_target)
            print('Random Forest - Test R^2:', np.round_(rf.score(test_input, test_target), 3))
            prediction = np.round_(rf.predict(test_input), 2)

            df_import = pd.DataFrame()
            df_import = df_import.append(pd.DataFrame([np.round((rf.feature_importances_) * 100, 2)],
                                                      columns=['probeId', 'pulseVoltage', 'numTxCycles',
                                                               'numTxElements', 'txFrequencyHz', 'elevAperIndex',
                                                               'isTxAperModulationEn', 'txpgWaveformStyle',
                                                               'scanRange', 'probePitchCm', 'probeRadiusCm',
                                                               'probeElevAperCm0', 'probeElevAperCm1',
                                                               'probeNumElements', 'probeElevFocusRangCm',
                                                               'probeType']), ignore_index=True)

            mae = mean_absolute_error(test_target, prediction)
            print('|(?????? - ?????????)|:', mae)
            print(df_import)


    except():
        print('error: func_machine_learning')


if __name__ == '__main__':
    server_address, ID, password, list_databases = func_conf_get()
    AOP_data = func_sql_get(server_address=server_address, ID=ID, password=password, list_databases=list_databases)
    print(len(AOP_data.index))
    data, target = func_preprocess(AOP_data=AOP_data)
    func_machine_learning(selected_ML='RandomForestRegressor', data=data, target=target)