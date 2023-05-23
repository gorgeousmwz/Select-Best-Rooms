import os
import pandas as pd

class DataManager:
    def __init__(self):
        self.data = pd.DataFrame()

    def read_csv_file(self, folder_path,year,fields=None):
        '''
        读取csv文件
        params:
            folder_path: 数据文件夹
            fields: 指定读取字段列表
            year: 读取哪一年的数据
        '''
        files = os.listdir(folder_path)
        csv_files = [file for file in files if file.endswith('.csv')] # 文件名列表
        file_path = os.path.join(folder_path, csv_files[int(year)-2014])
        df = pd.read_csv(file_path, usecols=fields) # 读取指定字段数据
        df.fillna('', inplace=True) # 处理空缺值
        self.data = df # 增添到数据集

    def read_csv_files(self, folder_path,fields=None,years=range(10)):
        '''
        读取多个csv文件
        params:
            folder_path: 数据文件夹
            fields: 指定读取字段列表
            years: 读取哪几年的数据
        '''
        files = os.listdir(folder_path)
        csv_files = [file for file in files if file.endswith('.csv')] # 文件名列表
        
        for i in years:
            file_path = os.path.join(folder_path, csv_files[i]) # 获取数据路径
            df = pd.read_csv(file_path, usecols=fields) # 读取指定字段数据
            df.fillna('', inplace=True) # 处理空缺值
            self.data = pd.concat([self.data,df], ignore_index=True) # 增添到数据集

if __name__=='__main__':
    # 创建DataManager对象
    manager = DataManager()
    # # 读取CSV文件夹
    manager.read_csv_files('D:\\本科\\时空数据分析与挖掘\\实习-mwz\\data_csv',['OBJECTID'],range(7,10))
    # # 打印数据
    print(manager.data)