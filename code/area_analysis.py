import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from data_manager import DataManager
from mpl_toolkits.mplot3d import Axes3D
import math
from prettytable import PrettyTable
import sys

# 绘制区域编号图
# 创建一个新的图形
fig, ax = plt.subplots(dpi=200)
# 读取区域数据并可视化
areas=gpd.read_file('data_airbnb\\areas.geojson') # 读取区域数据
areas.plot(ax=ax,cmap='twilight_shifted') # 可视化
# 在每个面区域上标注"area_id"属性值
for idx, row in areas.iterrows():
    # 添加注释标注
    plt.annotate(text=row['area_id'], xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                xytext=(0, 0), textcoords="offset points")
# 设置横轴和纵轴标签
ax.set_title('区域编号图')
ax.set_xlabel('经度')
ax.set_ylabel('纬度')
plt.savefig('result\\areas_analysis\\areas.png')


# 绘制crime、room在area上的图
# 创建一个新的图形
fig, ax = plt.subplots(dpi=200)
# 读取区域数据并可视化
areas=gpd.read_file('data_airbnb\\areas.geojson') # 读取区域数据
areas.plot(ax=ax,cmap='twilight_shifted') # 可视化
# 在每个面区域上标注"area_id"属性值
for idx, row in areas.iterrows():
    # 添加注释标注
    plt.annotate(text=row['area_id'], xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                xytext=(0, 0), textcoords="offset points")
# 读取犯罪点数据并可视化
manager=DataManager()
manager.read_csv_files('data_csv',years=range(7,10)) # 读取21 22 23年的数据
crimes_geometry = [Point(xy) for xy in zip(manager.data['LONGITUDE'], manager.data['LATITUDE'])] # 以经纬度为坐标转为点
crimes= gpd.GeoDataFrame(manager.data, geometry=crimes_geometry) # 生成geodataframe
crimes.plot(ax=ax,markersize=0.1,color='red') # 绘制
# 读取房屋数据并可视化
rooms_df = pd.read_csv('result\\review_analysis\\rooms_score.csv') # 读取房屋数据
rooms_df.fillna('', inplace=True) # 处理空缺值
rooms_geometry = [Point(xy) for xy in zip(rooms_df['longitude'], rooms_df['latitude'])] # 以经纬度为坐标转为点
rooms= gpd.GeoDataFrame(rooms_df, geometry=rooms_geometry) # 生成geodataframe
rooms.plot(ax=ax,markersize=0.1,color='green') # 绘制
# 设置横轴和纵轴标签
ax.set_title('犯罪&房源区域分布图')
ax.set_xlabel('经度')
ax.set_ylabel('纬度')
plt.savefig('result\\areas_analysis\\area_crime_room.png')


# 进行空间连接获得具有area_id的crimes点
crimes_area = gpd.sjoin(crimes, areas, how='left', op='within').drop(columns=['index_right','neighbourhood'])
crimes_area = crimes_area.groupby('CCN').first() # 剔除重复的
# 进行空间连接获得具有area_id的rooms点
rooms_area = gpd.sjoin(rooms, areas, how='left', op='within').drop(columns=['index_right','neighbourhood_right'])
rooms_area = rooms_area.groupby('id').first() # 剔除重复的
# # 存储经过区域分析的crimes和rooms
crimes_area.to_file('result\\areas_analysis\\crimes_area.geojson',driver='GeoJSON')
rooms_area.to_file('result\\areas_analysis\\rooms_area.geojson',driver='GeoJSON')


# 统计每个area中各有多少crime和room
count_crimes=[0]*39
count_rooms=[0]*39
# 遍历crimes_area中的每一行数据，计数
for index, row in crimes_area.iterrows():
    if not math.isnan(row['area_id']):
        count_crimes[int(row['area_id'])-1]+=1
# 遍历rooms_area中的每一行数据，计数
for index, row in rooms_area.iterrows():
    if not math.isnan(row['area_id']):
        count_rooms[int(row['area_id'])-1]+=1
# 输出结果
pt=PrettyTable()
pt.field_names=["Area_ID","Num of Crime","Num of Room"]
for i in range(1,40):
    pt.add_row([i,count_crimes[i-1],count_rooms[i-1]])
sys.stdout = open('result\\areas_analysis\\crime&room_count_in_area.txt', 'w')
print(pt)
sys.stdout.close()
# 存储为geojson
areas=areas.assign(count_crime=count_crimes)
areas=areas.assign(count_room=count_rooms)
areas.to_file('result\\areas_analysis\\areas_count.geojson',driver="GeoJSON")


# 绘crime制柱状图
data = {'area_id': list(range(1,40)), 'count_crime': count_crimes}
gdf = gpd.GeoDataFrame(data, geometry=areas.geometry)
# 创建立体图形
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')
# 绘制三维柱状图
for idx, row in gdf.iterrows():
    centroid = row.geometry.centroid
    x = centroid.x
    y = centroid.y
    z = row['count_crime']
    dx = dy = 0.01  # x 和 y 轴上柱状图的宽度
    dz = z  # 柱状图的高度（value）
    ax.bar3d(x, y, 0, dx, dy, dz)
    # 在柱状图上方标注值
    txt='area{}({})'.format(idx+1,z)
    ax.text(x + dx / 2, y + dy / 2, 1.05*dz,txt, ha='center', va='center')
# 设置坐标轴标签
ax.set_title('区域犯罪数量柱状图')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Num of crime')
# 显示图形
plt.savefig('result\\areas_analysis\\bar_of_crime_in_area.png')


# 绘制room柱状图
data = {'area_id': list(range(1,40)), 'count_room': count_rooms}
gdf = gpd.GeoDataFrame(data, geometry=areas.geometry)
# 创建立体图形
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')
# 绘制三维柱状图
for idx, row in gdf.iterrows():
    centroid = row.geometry.centroid
    x = centroid.x
    y = centroid.y
    z = row['count_room']
    dx = dy = 0.01  # x 和 y 轴上柱状图的宽度
    dz = z  # 柱状图的高度（value）
    ax.bar3d(x, y, 0, dx, dy, dz)
    # 在柱状图上方标注值
    txt='area{}({})'.format(idx+1,z)
    ax.text(x + dx / 2, y + dy / 2, 1.05*dz,txt, ha='center', va='center')
# 设置坐标轴标签
ax.set_title('区域房源数量柱状图')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Num of room')
# 显示图形
plt.savefig('result\\areas_analysis\\bar_of_room_in_area.png')