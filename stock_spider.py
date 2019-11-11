# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 16:20:52 2019

@author: ango
"""

import requests
from bs4 import BeautifulSoup as bs
import time
import random
import json
import numpy as np
import math
from bert_embedding import BertEmbedding


#获得股票代码函数
def getStockList(lst):

    #确定头文件内容
    headers = {
        'Referer': 'http://quote.eastmoney.com',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.81 Safari/537.36'
    }
    #确认stockURL
    stockURL = 'http://quote.eastmoney.com/usstocklist.html'
    #建立response
    response = requests.get(stockURL, headers=headers).content.decode('gbk')  # 网站编码为gbk 需要解码
    #初始化beautifulsoup4 以html.parser编码器处理 （该编码器纠错能力强
    soup = bs(response, 'html.parser')

    #获取quotesearch标签div下所有ul
    all_ul = soup.find('div', id='quotesearch').find_all('ul')  # 获取两个ul 标签数据
    #以utf-8编码创建txt以存放相关内容
    with open('stock_names.txt', 'w+', encoding='utf-8') as f:
        for ul in all_ul:   #有两个ul 需要遍历

            all_a = ul.find_all('a')  # 获取ul 下的所有的a 标签
            for a in all_a:     #有大量a标签
                if a['title'] != '':
                    title = a['title']
                    if a.text != '':
                        text = a.text.split('(')[1].split(')')[0]
                        # f.write(text + '\n')  #装入txt
                        lst.append({'name':title, 'short':text})      #装入list
    return lst

def make_sentence(word):
    rand = np.random.rand(9).tolist()

    actions = ['buy', 'sell']
    action = actions[round(rand[0])]

    file = open('stock_names.json', 'r')
    tickers = json.load(file)
    ticker = tickers[math.floor(rand[1] * len(tickers))]['short']
    ticker = word

    time_format = ["%Y/%m/%d", "%m/%d/%Y"]
    ticks = time.time()
    date = time.localtime(ticks * rand[2] + ticks)
    maturity = time.strftime(time_format[round(rand[3])], date)

    amount = str(round(rand[4] * 100) * pow(10, round(rand[5] * 10)))

    coupon = str(round(rand[6], 2))

    extras = ["can you let me know if you can ",
              "you guys know ",
              "Hi ",
              "want to ",
              "Hi, can u offer ",
              "Good morning, pls ",
              "can you offer ",
              "I can ",
              "can you ",
              "Talking to someone of "]
    extra = extras[math.floor(rand[7] * len(extras))]
    shuffle = [ticker, maturity, amount, coupon]
    random.shuffle(shuffle)
    string = extra + action
    for item in shuffle:
        string += " " + item

    return string

def main():
    #初始化列表
    slist = []
    # slist = getStockList(slist)
    # # getStockInfo(slist)
    # file = open('stock_names.json','w')
    # json.dump(slist, file, ensure_ascii=False)
    # file.close()

    words = ['all','bid','nyc','trade','care','btw','ice','delta','leg','see']
    negative_sentences = ['Lets have a party in , all of us',
                          'I bid on that game',
                          'just visited NYC and my girls loves it',
                          'Is the trade deal negotiation completed',
                          'why would someone even care for that trade we were discussing',
                          'work is done, btw, shall we go grab a beer?',
                          'not ice bond she just raised, she asks ice for her cocktail',
                          'delta team won the game',
                          'set your leg on the way to learning stuff',
                          'i see fire inside the mountains']
    vec = []
    # file = open("sentences.txt",'w')
    bert_embedding = BertEmbedding()
    for i in range(len(words)):
        sentences = []
        for j in range(20):
            sentence = make_sentence(words[i])
            # file.write(sentence+'\n')
            sentences.append(sentence)
        sentences.append(negative_sentences[i])
        result = bert_embedding(sentences)

        pos_vec = [result[k][1][result[k][0].index(words[i])].tolist() for k in range(len(result)-1) if words[i] in result[k][0]]

        neg_vec = [result[-1][1][result[-1][0].index(words[i])].tolist()]
        vec.append({'words':words[i],'pos':pos_vec,'neg':neg_vec,'sentence':sentences})
    file = open('vec.json','w')
    json.dump(vec,file)
    file.close()
    # file.close()
    # file = open("sentences.json",'w')
    # json.dump(sentences, file)
    # file.close()

# main()
file = open('vec.json','r')
vecs = json.load(file)
word = 'all'
sentence = ['We all have a meeting in 19:30, room 305']
bert_embedding = BertEmbedding()
result = bert_embedding(sentence)
word_vec = result[0][1][result[0][0].index(word)]
for item in vecs:
    if item['words'] == word:
        dict = item
for i,pos_vec in enumerate(dict['pos']):
    print("distance of positive: ",np.linalg.norm(np.array(pos_vec) - word_vec),dict['sentence'][i])
print("distance of negative: ",np.linalg.norm(np.array(dict['neg'][0]) - word_vec),dict['sentence'][-1])