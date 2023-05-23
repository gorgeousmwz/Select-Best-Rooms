import pandas as pd
import matplotlib.pyplot as plt
from data_manager import DataManager
import numpy as np

# 读取数据
manager=DataManager()
manager.read_csv_files('D:\\本科\\时空数据分析与挖掘\\实习-mwz\\data_csv',['REPORT_DAT'])
def showNumOfCrimesInPeriod(period):
    '''统计每个相同时间单位的数据条目数量,并绘制成图'''
    n=0
    size=(0,0)
    dpi=300
    if period=='year':
        n=4
        size=(10,8)
    elif period=='month':
        n=7
        size=(22,7)
    elif period=='day':
        n=10
        size=(500,5)
        dpi=100
    else:
        return
    # 统计每个相同时间单位的数据条目数量
    manager.data['REPORT_DAT'] = manager.data['REPORT_DAT'].str[:n]
    counts = manager.data['REPORT_DAT'].value_counts().sort_index()
    fig=plt.figure(figsize=size)
    # 绘制柱状图
    counts.plot(kind='bar', xlabel=period, ylabel='num', title=f'Number of crime in every {period}',colormap=plt.cm.get_cmap('cool'))
    # 在每个柱的顶部添加文本标签
    for i, value in enumerate(counts):
        plt.text(i, value, str(value), ha='center', va='bottom')
    # 显示图形
    plt.savefig(f'result\\crime_analysis\\crime_num_in_{period}.png',dpi=dpi)

if __name__=='__main__':
    showNumOfCrimesInPeriod('day')

