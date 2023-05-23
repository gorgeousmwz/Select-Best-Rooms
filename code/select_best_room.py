import geopandas as gpd
import matplotlib.pyplot as plt
import random
import math
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
from geopy import distance
from prettytable import PrettyTable
from sklearn.preprocessing import MinMaxScaler
import sys


def generate_dangerous_coefficient(gender=0,car=0,rich=0):
    '''
        根据犯罪的轻重(判期长短)，生成危险系数(会根据游客的特殊性进行变化)
        女性性虐待系数变大
        自驾游车辆盗窃系数变大
        带有贵重物品的盗窃抢劫等系数变大
    '''
    #个性化参数
    additional_gender=max(5*gender,1)
    additional_car=max(2*car,1)
    additional_rich=max(1.5*rich,1)
    # 危险系数
    THEFT_OTHER=random.uniform(0.1,2)*additional_rich
    THEFT_AUTO=random.uniform(10,15)*additional_car
    MOTOR_VEHICLE_THEFT=random.uniform(5,15)*additional_car
    BURGLARY=random.uniform(5,15)*additional_rich
    ARSON=random.uniform(10,30)
    ASSAULT_DANGEROUS_WEAPON=random.uniform(10,30)
    HOMICIDE=1000
    SEX_ABUSE=random.uniform(5,15)*additional_gender
    ROBBERY= random.uniform(5,15)*additional_car
    dangerous_coefficient={
        'THEFT/OTHER':THEFT_OTHER,
        'THEFT F/AUTO':THEFT_AUTO,
        'MOTOR VEHICLE THEFT':MOTOR_VEHICLE_THEFT,
        'BURGLARY':BURGLARY,
        'ARSON':ARSON,
        'ASSAULT W/DANGEROUS WEAPON':ASSAULT_DANGEROUS_WEAPON,
        'HOMICIDE':HOMICIDE,
        'SEX ABUSE':SEX_ABUSE,
        'ROBBERY':ROBBERY
    }
    return dangerous_coefficient

def drawBarDangerousArea(dangerous_in_area,dangerous_rate_in_area,safe_value_threshold,safe_rate_threshold):
    '''
        绘制每个区域的危险数值和危险率的柱状图
    '''
    # 绘制成图
    x_index=list(range(1,40))
    bar_width = 0.25 # 设置柱子的宽度
    # 生成x轴位置
    bar1_pos = np.arange(len(x_index))
    bar2_pos = [x + bar_width for x in bar1_pos]
    # 创建图形和第一个坐标轴
    fig, ax1 = plt.subplots(dpi=200)
    bar1 = ax1.bar(bar1_pos, dangerous_in_area, bar_width, color='red', label='dangerous value')
    # 设置第一个坐标轴标签和标题
    ax1.set_xlabel('area id')
    ax1.set_ylabel('dangerous value', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_title('危险地区分析图')
    # 创建第二个坐标轴
    ax2 = ax1.twinx()
    bar2 = ax2.bar(bar2_pos, dangerous_rate_in_area, bar_width, color='yellow', label='dangerous rate')
    # 设置第二个坐标轴标签
    ax2.set_ylabel('dangerous rate', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    # 设置x轴刻度
    ax1.set_xticks(bar1_pos + bar_width / 2)
    ax1.set_xticklabels(x_index,fontsize=6)
    # 绘制安全线
    line1=ax1.axhline(y=safe_value_threshold, color='blue', linestyle='--', label='safe value threshold')
    line2=ax2.axhline(y=safe_rate_threshold, color='green', linestyle='--', label='safe rate threshold')
    # 添加图例
    draws = [bar1, bar2, line1, line2]
    labels = [draw.get_label() for draw in draws]
    ax1.legend(draws, labels)
    # 自动调整布局
    fig.tight_layout()
    # 显示图形
    plt.savefig('result\\select_best_room\\dangerous_value&rate_every_area.png')

def labelSafeArea(areas):
    '''
    标注危险区域
    '''
    # 创建一个新的图形
    fig, ax = plt.subplots(dpi=300)
    areas.plot(column='issafe', cmap=ListedColormap(['red','green'])) # 可视化
    # 在每个面区域上标注"area_id"属性值
    for idx, row in areas.iterrows():
        # 添加注释标注
        plt.annotate(text=row['area_id'], xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                    xytext=(0, 0), textcoords="offset points")
    # 设置横轴和纵轴标签
    ax.set_xlabel('经度')
    ax.set_ylabel('纬度')
    ax.set_title('较安全的区域标注图')
    plt.savefig('result\\select_best_room\\areas_labeled_safe.png')

def calDistance(point1,point2):
    '''计算两点之间的距离'''
    return distance.distance(point1, point2).km

def drawBoxCheapRoom(prices,price_threshold):
    '''绘制房价的箱型图，标志出价格阈值'''
    pt=PrettyTable()
    pt.field_names=['Room Type','Average','Min','Max','Variance']
    pt.add_row(['Entire home/apt',np.average(prices['Entire home/apt']),min(prices['Entire home/apt']),max(prices['Entire home/apt']),np.var(prices['Entire home/apt'])])
    pt.add_row(['Private room',np.average(prices['Private room']),min(prices['Private room']),max(prices['Private room']),np.var(prices['Private room'])])
    pt.add_row(['Shared room',np.average(prices['Shared room']),min(prices['Shared room']),max(prices['Shared room']),np.var(prices['Shared room'])])
    pt.add_row(['Hotel room',np.average(prices['Hotel room']),min(prices['Hotel room']),max(prices['Hotel room']),np.var(prices['Hotel room'])])
    pt.add_row(['SUM',np.average(prices['SUM']),min(prices['SUM']),max(prices['SUM']),np.var(prices['SUM'])])
    sys.stdout = open('result\\select_best_room\\room_price.txt', 'w')
    print(pt)
    sys.stdout.close()
    # 创建一个箱型图
    fig, ax = plt.subplots(dpi=200)
    # 将四个类别的价格数据放在一个列表中
    data = [prices['Entire home/apt'],prices['Private room'],prices['Shared room'],prices['Hotel room'],prices['SUM']]
    # 设置箱型图的颜色
    boxprops = dict(color='black')
    # 绘制箱型图
    ax.boxplot(data, labels=['Entire', 'Private', "Shared", "Hotel", "SUM"], vert=False, boxprops=boxprops,showfliers=False)
    # 绘制四种类别的点
    colors = ['pink', 'blue', 'purple', 'yellow','red']
    for i in range(len(data)):
        ax.scatter(data[i], [i+1] * len(data[i]), color=colors[i],s=2,edgecolors=colors[i])
    # 绘制价格阈值直线
    ax.axvline(price_threshold, color='green', linestyle='--', label='price threshold')
    # 设置x轴刻度截断在0-1000
    ax.set_xlim(0, 1000)
    # 设置图形标题和纵轴标签
    ax.set_title('房源价格分布图')
    ax.set_xlabel('price')
    # 显示图例
    ax.legend()
    # 显示图形
    plt.savefig('result\\select_best_room\\room_price.png')

def labelCheapRoom(areas,rooms):
    '''标注低于价格阈值的房源'''
    # 创建一个新的图形
    fig, ax = plt.subplots(dpi=200)
    # 读取区域数据并可视化
    areas.plot(ax=ax) # 可视化
    # 在每个面区域上标注"area_id"属性值
    for idx, row in areas.iterrows():
        # 添加注释标注
        plt.annotate(text=row['area_id'], xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                    xytext=(0, 0), textcoords="offset points")
    rooms.plot(ax=ax,markersize=0.1,column='ischeap', cmap=ListedColormap(['red','green'])) # 绘制
    # 设置横轴和纵轴标签
    ax.set_xlabel('经度')
    ax.set_ylabel('纬度')
    ax.set_title('实惠房源标注图')
    plt.savefig('result\\select_best_room\\rooms_labeled_cheap.png')

def labelCloseRoom(areas,rooms,destination):
    '''标注距离目的地较近的房源'''
    # 创建一个新的图形
    fig, ax = plt.subplots(dpi=200)
    # 读取区域数据并可视化
    areas.plot(ax=ax) # 可视化
    # 在每个面区域上标注"area_id"属性值
    for idx, row in areas.iterrows():
        # 添加注释标注
        plt.annotate(text=row['area_id'], xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                    xytext=(0, 0), textcoords="offset points")
    rooms.plot(ax=ax,markersize=0.1,column='isclose', cmap=ListedColormap(['red','green'])) # 绘制
    # 绘制目的地
    ax.scatter(destination[1], destination[0], marker='p', color='purple', s=100)
    # 设置横轴和纵轴标签
    ax.set_xlabel('经度')
    ax.set_ylabel('纬度')
    ax.set_title('邻近房源标注图')
    plt.savefig('result\\select_best_room\\rooms_labeled_close.png')

def selectSafeArea(areas,rooms,crimes,safe_value_threshold,safe_rate_threshold,gender=0,car=0,rich=0):
    '''筛选出较为安全的区域'''
    dangerous_in_area=[0]*39 # 每个区域中的危险数值
    dangerous_rate_in_area=[0]*39 # 每个区域的危险率
    num_repetition=10 # 重复计算次数
    for time in range(num_repetition): # 计算n次取均值
        print(time)
        dangerous_coefficient=generate_dangerous_coefficient(gender,car,rich) # 获取危险系数
        for index,row in crimes.iterrows(): # 遍历每个犯罪案件
            type=row['OFFENSE'] # 犯罪类型
            area_id=row['area_id'] # 犯罪区域
            if math.isnan(area_id) == False:
                dangerous_in_area[int(area_id)-1]+=dangerous_coefficient[type] # 累计危险数值
    dangerous_in_area=list(map(lambda x:x/num_repetition,dangerous_in_area)) # 求平均
    # 计算危险率
    for area_id in range(1,40):
        dangerous_rate_in_area[area_id-1]=\
            dangerous_in_area[area_id-1]/areas.loc[areas['area_id'] == area_id, 'count_crime'].values[0]
    # 绘制柱状图
    drawBarDangerousArea(dangerous_in_area,dangerous_rate_in_area,safe_value_threshold,safe_rate_threshold)
    # 标注危险区域
    flag=[True]*39
    for i in range(39):
        if dangerous_in_area[i]>safe_value_threshold or dangerous_rate_in_area[i]>safe_rate_threshold: # 如果有一个指标超出阈值
            flag[i]=False # 定义为不安全
    areas=areas.assign(dangerous_value=dangerous_in_area) # 添加危险值列
    areas=areas.assign(dangerous_rate=dangerous_rate_in_area) # 添加危险率列
    areas=areas.assign(issafe=flag) # 添加安全与否标志列
    areas.to_file('result\\select_best_room\\areas_labeled_safe.geojson',driver='GeoJSON') # 存储为geojson
    labelSafeArea(areas) # 标注安全区域
    # 在rooms上也标记上issafe
    for index,row in rooms.iterrows():
        rooms.at[index,'issafe']=flag[int(row['area_id'])-1]
    return areas,rooms

def selectCheapAndCloseRoom(areas,rooms,destination,price_threshold,distance_threshold):
    '''筛选出较实惠且距离目的地较近的房源'''
    prices={
        'Entire home/apt':[],
        'Private room':[],
        'Shared room':[],
        'Hotel room':[],
        'SUM':[]
    }
    for index,row in rooms.iterrows(): # 遍历每个房源
        type=row['room_type'] # 获取房屋类型
        price=row['price'] # 获取价格
        prices[type].append(price)
        prices['SUM'].append(price)
        if price>price_threshold: # 价格筛选
            rooms.at[index,'ischeap']=False # 进行标志
        else:
            rooms.at[index,'ischeap']=True
        dis=calDistance((row['latitude'],row['longitude']),destination) # 计算该房源距离目的地的距离
        rooms.at[index,'distance']=dis # 记录距离
        if dis > distance_threshold: # 距离筛选
            rooms.at[index,'isclose']=False
        else:
            rooms.at[index,'isclose']=True
    rooms.to_file('result\\select_best_room\\rooms_labeled_safe&cheap&close.geojson',driver='GeoJSON')
    drawBoxCheapRoom(prices,price_threshold) # 绘制箱型图
    labelCheapRoom(areas,rooms) # 标注实惠房源
    labelCloseRoom(areas,rooms,destination) # 标注距离较近的房源
    return rooms

def drawBoxFES(rooms):
    '''绘制最终得分的箱型图'''
    feses={'Entire home/apt':[],'Private room':[],'Shared room':[],'Hotel room':[],'SUM':[]}
    for index,room in rooms.iterrows():
        fes=room['FES']
        type=room['room_type']
        feses[type].append(fes)
        feses['SUM'].append(fes)
    pt=PrettyTable()
    pt.field_names=['Room Type','Average','Min','Max','Variance']
    pt.add_row(['Entire home/apt',np.average(feses['Entire home/apt']),min(feses['Entire home/apt']),max(feses['Entire home/apt']),np.var(feses['Entire home/apt'])])
    pt.add_row(['Private room',np.average(feses['Private room']),min(feses['Private room']),max(feses['Private room']),np.var(feses['Private room'])])
    pt.add_row(['Shared room',np.average(feses['Shared room']),min(feses['Shared room']),max(feses['Shared room']),np.var(feses['Shared room'])])
    pt.add_row(['Hotel room',np.average(feses['Hotel room']),min(feses['Hotel room']),max(feses['Hotel room']),np.var(feses['Hotel room'])])
    pt.add_row(['SUM',np.average(feses['SUM']),min(feses['SUM']),max(feses['SUM']),np.var(feses['SUM'])])
    sys.stdout = open('result\\select_best_room\\room_fes.txt', 'w')
    print(pt)
    sys.stdout.close()
    # 创建一个箱型图
    fig, ax = plt.subplots(dpi=200)
    # 将5个类别的价格数据放在一个列表中
    data = [feses['Entire home/apt'],feses['Private room'],feses['Shared room'],feses['Hotel room'],feses['SUM']]
    # 设置箱型图的颜色
    boxprops = dict(color='black')
    # 绘制箱型图
    ax.boxplot(data, labels=['Entire', 'Private', "Shared", "Hotel", "SUM"], vert=False, boxprops=boxprops,showfliers=False)
    # 绘制四种类别的点
    colors = ['green', 'blue', 'purple', 'yellow','red']
    for i in range(len(data)):
        ax.scatter(data[i], [i+1] * len(data[i]), color=colors[i],s=2,edgecolors=colors[i])
    # 设置图形标题和纵轴标签
    ax.set_title('房源最终得分分布图')
    ax.set_xlabel('FES')
    # 显示图形
    plt.savefig('result\\select_best_room\\room_fes.png')

def saveBestRooms(best_rooms):
    '''存储最好的房源'''
    best_rooms.to_file('result\\select_best_room\\best_rooms.geojson',driver='GeoJSON')
    sys.stdout = open('result\\select_best_room\\best_rooms.txt', 'w',encoding='utf-8')
    print('The general information of best rooms is follows.')
    pt=PrettyTable()
    pt.field_names=['id','dangerous value','dangerous rate','price','distance','review score','FES']
    for index,room in best_rooms.iterrows():
        pt.add_row([room['id'],room['dangerous_value'],room['dangerous_rate'],room['price'],room['distance'],room['score'],room['FES']])
    print(pt)
    print('The gdetail information of best rooms is follows.')
    index=0
    rooms_detail=pd.read_csv('data_airbnb\\rooms_detail.csv')
    for _,room in best_rooms.iterrows():
        print('The room ranked {} is {}'.format(index+1,room['id']))
        pt=PrettyTable()
        pt.field_names=['Key','Description']
        keys=best_rooms.columns.tolist()
        for key in keys:
            pt.add_row([key,room[key]])
        detail=rooms_detail[rooms_detail['id']==room['id']] # 获取对应id的行
        pt.add_row(['description',detail['description']])
        pt.add_row(['neighborhood_overview',detail['neighborhood_overview']])
        pt.add_row(['listing_url',detail['listing_url']])
        pt.add_row(['picture_url',detail['picture_url']])
        pt.add_row(['host_url',detail['host_url']])
        print(pt)
        index+=1
    sys.stdout.close()

def labelBestRooms(areas,best_rooms):
    '''
    标注危险区域
    '''
    # 创建一个新的图形
    fig, ax = plt.subplots(dpi=200)
    # 读取区域数据并可视化
    areas.plot(ax=ax,column='issafe', cmap=ListedColormap(['red','blue'])) # 可视化 # 可视化
    # 在每个面区域上标注"area_id"属性值
    for idx, row in areas.iterrows():
        # 添加注释标注
        plt.annotate(text=row['area_id'], xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                    xytext=(0, 0), textcoords="offset points")
    # 绘制目的地
    ax.scatter(destination[1], destination[0], marker='p', color='purple', s=100)
    # 绘制推荐房源
    for index,best_room in best_rooms.iterrows():
        lon=best_room['geometry'].x
        lat=best_room['geometry'].y
        ax.scatter(lon, lat, marker='^', color='green', s=10)
    # 设置横轴和纵轴标签
    ax.set_xlabel('经度')
    ax.set_ylabel('纬度')
    ax.set_title('推荐房源标志图')
    plt.savefig('result\\select_best_room\\rooms_labeled_best.png')

def selectBestRooms(areas,rooms):
    '''选择优秀房源进行推荐'''
    # 筛选剩下的房源，更具危险性、价格、距离选取最好的房源
    rooms_temp=rooms.copy()
    areas_temp=areas.copy()
    # 归一化到[0,1]
    scaler = MinMaxScaler()
    areas_temp['dangerous_value'] = scaler.fit_transform(areas_temp[['dangerous_value']])
    areas_temp['dangerous_rate'] = scaler.fit_transform(areas_temp[['dangerous_rate']])
    rooms_temp['price'] = scaler.fit_transform(rooms_temp[['price']])
    rooms_temp['distance'] = scaler.fit_transform(rooms_temp[['distance']])
    rooms_temp['score'] = rooms_temp['score'].replace('', np.nan) # 将空字符串替换为 NaN
    non_empty_values = rooms_temp['score'].dropna().values.reshape(-1, 1) # 创建一个副本，只包含非空值的子集
    normalized_values = scaler.fit_transform(non_empty_values) # 对非空值进行归一化
    rooms_temp.loc[rooms_temp['score'].notnull(), 'score'] = normalized_values # 将归一化后的值赋回原 DataFrame 的相应列
    # 遍历每个房源计算最后的总评分值
    for index,room in rooms_temp.iterrows():
        dangerous_value=float(areas_temp.loc[areas_temp['area_id']==room['area_id'],'dangerous_value']) # 获取危险值
        dangerous_rate=float(areas_temp.loc[areas_temp['area_id']==room['area_id'],'dangerous_rate']) # 获取危险率
        price=room['price'] # 获取价格
        dis=room['distance'] # 获取距离
        score=room['score'] # 获取评分
        if math.isnan(score):
            score=0
        fes = - dangerous_value - dangerous_rate - price - dis + score # 最终评价得分Final evaluation score
        rooms.at[index,'FES']=fes
    rooms['FES']=scaler.fit_transform(rooms[['FES']])*100 # 总评分归一化
    drawBoxFES(rooms) # 绘制最终得分箱型图
    rooms.to_file('result\\select_best_room\\rooms_labeled_FES.geojson',driver='GeoJSON')
    # 筛选出既满足阈值条件，FES评分前十的房源最为推荐房源
    for index,room in rooms.iterrows():
        issafe=int(room['issafe']) # 是否安全
        ischeap=int(room['ischeap']) # 是否便宜
        isclose=int(room['isclose']) # 是否邻近目的地
        dangerous_value=float(areas.loc[areas['area_id']==room['area_id'],'dangerous_value']) # 获取危险值
        dangerous_rate=float(areas.loc[areas['area_id']==room['area_id'],'dangerous_rate']) # 获取危险率
        rooms.at[index,'FES']=issafe*ischeap*isclose*room['FES'] # 剔除不满足阈值条件的房源
        rooms.at[index,'dangerous_value']=dangerous_value
        rooms.at[index,'dangerous_rate']=dangerous_rate
    best_rooms = rooms.nlargest(10, 'FES') # 10个最好的房源
    saveBestRooms(best_rooms) # 存储最好的房源
    labelBestRooms(areas,best_rooms) # 标志最好房源在区域上

def excute(gender,car,safe_value_threshold,safe_rate_threshold,price_threshold,distance_threshold,destination):
    '''执行算法'''
    # 数据读取
    areas=gpd.read_file('result\\areas_analysis\\areas_count.geojson')
    crimes=gpd.read_file('result\\areas_analysis\\crimes_area.geojson')
    rooms=gpd.read_file('result\\areas_analysis\\rooms_area.geojson')
    # 筛选出较安全的房源
    areas,rooms=selectSafeArea(areas,rooms,crimes,safe_value_threshold,safe_rate_threshold,gender,car,rich)
    # 筛选出较便宜的房屋和距离目的地较近的房源
    rooms=selectCheapAndCloseRoom(areas,rooms,destination,price_threshold,distance_threshold)
    # 在筛选出的房源中根据FES得分推荐10所房源
    # # areas=gpd.read_file('result\\select_best_room\\areas_labeled_safe.geojson')
    # rooms=gpd.read_file('result\\select_best_room\\rooms_labeled_safe&cheap&close.geojson')
    selectBestRooms(areas,rooms)

gender=1
car=1
rich=1
safe_value_threshold=40000 # 安全值阈值
safe_rate_threshold=20 # 安全率阈值
price_threshold=100 # 价格阈值
distance_threshold=5 # 距离阈值(km)
destination=(38.88981295000001,-77.00902077737487) # 国会大厦

excute(gender,car,safe_value_threshold,safe_rate_threshold,price_threshold,distance_threshold,destination)






