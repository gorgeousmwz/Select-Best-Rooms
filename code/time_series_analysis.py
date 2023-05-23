from data_manager import DataManager
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse
import pandas as pd 
import matplotlib.pyplot as plt
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot

def timeDecompose(year):
    '''
    时序分解
        季节性分量：显示时间序列数据在每个季节周期中的重复模式。
        趋势分量：显示时间序列数据中的长期趋势或变化方向。
        残差分量：表示无法由季节性和趋势性解释的剩余部分，可以视为随机波动或噪声。
    params:
        year: 分解哪一年的数据
    '''
    # 数据读取
    manager=DataManager()
    manager.read_csv_file('D:\\本科\\时空数据分析与挖掘\\实习-mwz\\data_csv',year,['REPORT_DAT'])
    # 将REPORT_DAT列转换为日期时间类型
    manager.data['REPORT_DAT'] = pd.to_datetime(manager.data['REPORT_DAT'])
    # 创建新的DataFrame
    new= pd.DataFrame()
    # 提取日期列
    new['date'] = pd.to_datetime(manager.data['REPORT_DAT'].dt.strftime('%Y/%m/%d'))
    # 计算每个月份内的数据条数
    grouped_df= new.groupby('date').size().reset_index(name='Count')
    grouped_df=grouped_df.set_index('date') # 索引设为date
    # 对数据进行季节性分解
    # Multiplicative Decomposition 多项式（multiplicative）模型进行分解
    result_mul = seasonal_decompose(grouped_df['Count'], model='multiplicative', extrapolate_trend='freq')
    # Additive Decomposition 加法（additive）模型进行分解
    result_add = seasonal_decompose(grouped_df['Count'], model='additive', extrapolate_trend='freq')
    # Plot
    plt.rcParams.update({'figure.figsize': (12,12)})
    result_mul.plot().suptitle('Multiplicative Decompose', fontsize=16) # 绘制
    plt.tight_layout()  # 调整子图布局，避免重叠
    plt.savefig('result\\crime_analysis\\time_decompose\\{}_Multiplicative_Decompose.png'.format(year)) # 存储
    result_add.plot().suptitle('Additive Decompose', fontsize=16) # 绘制
    plt.tight_layout()  # 调整子图布局，避免重叠
    plt.savefig('result\\crime_analysis\\time_decompose\\{}_Additive_Decompose.png'.format(year)) # 存储
    # 根据季节分量、趋势分量、残差分量重建出原始值，原始值=season*tread*residual
    df_reconstructed = pd.concat([result_mul.seasonal, result_mul.trend, result_mul.resid, result_mul.observed], axis=1)
    df_reconstructed.columns = ['seas', 'trend', 'resid', 'actual_values']
    print(df_reconstructed.head())

def arima(year):
    '''
    ARIMA预测
    params:
        year: 分解哪一年的数据
    '''
    # 数据读取
    manager=DataManager()
    manager.read_csv_file('D:\\本科\\时空数据分析与挖掘\\实习-mwz\\data_csv',year,['REPORT_DAT'])
    # 将REPORT_DAT列转换为日期时间类型
    manager.data['REPORT_DAT'] = pd.to_datetime(manager.data['REPORT_DAT'])
    # 创建新的DataFrame
    new= pd.DataFrame()
    # 提取日期列
    new['date'] = pd.to_datetime(manager.data['REPORT_DAT'].dt.strftime('%Y/%m/%d'))
    # 计算每个月份内的数据条数
    grouped_df= new.groupby('date').size().reset_index(name='Count')
    grouped_df=grouped_df.set_index('date') # 索引设为date
    # fit model
    model = ARIMA(grouped_df['Count'], order=(5,1,0)) # 创建一个 ARIMA 模型对象,其中 5 是自回归阶数，1 是差分阶数，0 是移动平均阶数。
    model_fit = model.fit() # 对 ARIMA 模型进行拟合，估计模型的参数
    print(model_fit.summary()) # 打印拟合后的模型的摘要统计信息，包括参数估计、标准误差、置信区间等
    # plot residual errors
    residuals = DataFrame(model_fit.resid) # 存储拟合模型的残差项
    residuals.plot() # 绘制残差项的折线图，显示残差项随时间的变化
    #pyplot.show()
    residuals.plot(kind='kde') # 绘制残差项的核密度估计图，显示残差项的分布情况
    #pyplot.show()
    print(residuals.describe()) # 打印残差项的统计摘要，包括均值、标准差、最小值、最大值

    # 获取模型的预测结果
    forecast_values= model_fit.forecast(steps=20)  # 假设要预测未来20天的值
    # 绘制预测结果
    plt.figure()
    plt.plot(grouped_df.index, grouped_df['Count'], label='Actual')
    plt.plot(forecast_values.index, forecast_values, label='Forecast')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig('result\\crime_analysis\\time_prediction\\time_prediction\\{}_Arima_Prediction.png'.format(year))

# 分解2014-2023每年的crime数据
for year in range(2014,2024):
    timeDecompose(year)
    arima(year)