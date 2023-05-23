import pandas as pd
from tqdm import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer
import re
from gensim import corpora,models
from wordcloud import WordCloud
import pickle
import matplotlib.pyplot as plt
import math
from prettytable import PrettyTable
import numpy as np
import seaborn as sns
import sys

def read_data():
    '''数据读取'''
    reviews = pd.read_csv('data_airbnb\\reviews.csv', usecols=['listing_id','comments']) # 读取指定字段数据
    reviews.fillna('', inplace=True) # 处理空缺值
    return reviews.sample(100000)

def get_sentiment_score(text):
    '''
    计算compound"情感得分
    >0: 积极的
    <0: 消极的
    =0: 中性的
    '''
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores['compound']

def createWordCloud(lda_model,image_path):
    '''
    绘制词云
    params:
        lda_model: 训练好的lda模型
        image_path: 词云图片保存路径
    '''
    fig, axs = plt.subplots(ncols=2, nrows=math.ceil(lda_model.num_topics/2), figsize=(16,20))
    axs = axs.flatten()

    def color_func(word, font_size, position, orientation, random_state, font_path):
        return 'darkturquoise'

    for i, t in enumerate(range(lda_model.num_topics)):

        x = dict(lda_model.show_topic(t, 30))
        im = WordCloud(
            background_color='black',
            color_func=color_func,
            max_words=4000,
            width=300, height=300,
            random_state=0,
            font_path='C:\\ProgramData\\truetype\\arial\\arial.ttf'
        ).generate_from_frequencies(x)
        axs[i].imshow(im.recolor(colormap= 'Paired_r' , random_state=244), alpha=0.98)
        axs[i].axis('off')
        axs[i].set_title('Topic '+str(t))
    # vis
    plt.tight_layout()
    plt.savefig(image_path)# 保存图片

def data_processe(reviews):
    '''数据预处理'''
    scores ={} # 房屋得分统计
    database=[] # 总数据库
    for index,row in tqdm(reviews.iterrows(), total=len(reviews)): # 遍历每条评论
        doc=[]
        listing_id=row['listing_id'] # 获取评价房源对象id
        comments=re.sub(r'[^\w\s]', '', row['comments'])
        comments=comments.lower().split(' ') # 获取评价内容
        for word in comments:
            score=get_sentiment_score(word) # 计算得分
            scores[listing_id]=scores.get(listing_id,0)+score # 统计得分
            if score != 0: # 如果不是中性词，则纳如文档
                doc.append(word)
        database.append(doc)
    return scores,database

def drawScore(values):
    '''绘制Score'''
    pt=PrettyTable()
    pt.field_names=['Name','Average','Min','Max','Variance']
    pt.add_row(['Room Scores',np.average(values),min(values),max(values),np.var(values)])
    sys.stdout = open('result\\review_analysis\\scores.txt', 'w')
    print(pt)
    sys.stdout.close()
    # 绘制直方图
    plt.hist(values, bins=40)  # 设置bins参数来控制区间数量
    plt.title('房源评价得分直方图')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.savefig('result\\review_analysis\\room_scores_hist.png')
    # 绘制饼图
    fig, ax = plt.subplots(dpi=200)
    bins = [-float('inf'), 0, 50, 200, float('inf')] # 分组区间
    labels = ['<0:评分很差', '[0,50)]:评分一般', '[50,200):评分较好', '>=200:评分极好'] # 分组标签
    count, _ = np.histogram(values, bins=bins) # 统计每个区间的数量
    plt.pie(count, labels=labels, autopct='%1.1f%%', colors=['blue', 'green', 'yellow', 'red']) # 绘制饼状图
    plt.legend() # 添加图例
    plt.title('房源评价得分饼状图')# 添加标题
    plt.savefig('result\\review_analysis\\room_scores_pie.png')# 保存图形

if __name__=="__main__":
    # 读取数据并处理
    reviews=read_data()
    scores,database=data_processe(reviews)
        # 保存得分
    with open('result\\review_analysis\\topic_mining\\scores.pkl', 'wb') as f:
            pickle.dump(scores, f)
    # 保存清洗后的数据库
    with open('result\\review_analysis\\topic_mining\\review_database.pkl', 'wb') as f:
            pickle.dump(database, f)
    # 绘制
    with open('result\\review_analysis\\topic_mining\\scores.pkl', 'rb') as f: # 从文件中加载database变量
        scores = pickle.load(f)
    drawScore(list(scores.values()))
    # 更新rooms文件
    rooms=pd.read_csv('data_airbnb\\rooms.csv')
    for key,value in scores.items():
        rooms.loc[rooms['id']==key,'score']=value
    rooms.to_csv('result\\review_analysis\\rooms_score.csv',index=False)
    # 进行主题挖掘
    dictionary=corpora.Dictionary(database) # 生成字典
    dictionary.save_as_text('result\\review_analysis\\topic_mining\\dictionary.txt') # 保存字典
    # 生成词袋bag of word    
    corpus=[dictionary.doc2bow(text) for text in database]
    with open('result\\review_analysis\\topic_mining\\corpus.pkl', 'wb') as f:
            pickle.dump(corpus, f) # 保存corpus
    # 使用 LDA 模型进行主题分析
    lda_model = models.LdaModel(
        corpus, # 词袋
        num_topics=6, # 挖掘出的主题个数
        id2word=dictionary, # 字典 
        passes=10) # 训练过程中穿过语料库的次数
    lda_model.save('result\\review_analysis\\topic_mining\\lda_model.model') # 保存模型
    # 保存主题挖掘结果
    with open('result\\review_analysis\\topic_mining\\topics.txt', 'w') as f:
        # 循环写入每个主题下的词语
        for topic in lda_model.print_topics():
            f.write(str(topic) + '\n')