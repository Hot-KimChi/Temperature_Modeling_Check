import tkinter
from tkinter import *
from tkinter import ttk
from tkinter import filedialog

import configparser

import numpy as np
import pandas as pd
import pymssql


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
                    ORDER BY 1
                    '''

            Raw_data = pd.read_sql(sql=query, con=conn)
            # AOP_data = Raw_data.dropna()
            Raw_data.insert(0, "Database", f'{database}', True)
            AOP_data = Raw_data.append(Raw_data, ignore_index=True)                                                     ## DataFrame append할 경우, 동일한 parameter 갯수. // ignore_index(True) 인덱스가 기존해의 뒷 번호로 지정

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


if __name__ == '__main__':
    server_address, ID, password, list_databases = func_conf_get()
    AOP_data = func_sql_get(server_address=server_address, ID=ID, password=password, list_databases=list_databases)
    print(AOP_data.head())
    print(len(AOP_data.index))