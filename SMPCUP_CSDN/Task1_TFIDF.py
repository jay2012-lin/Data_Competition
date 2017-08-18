# -*-coding:UTF-8-*-
# Author:jaylin
# File:Task1_TFIDF.py
# Time:2017/7/24 9:22
import numpy as np
import datetime
import jieba
import pickle
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

text_rank_result = 'all_blog_themes.pickle'
train_blog_file = unicode('../data/SMP/SMPCUP2017任务1训练集/SMPCUP2017_TrainingData_Task1.txt','utf-8')
valid_blog_file = unicode('../data/SMPCUP2017验证集/SMPCUP2017_ValidationSet_Task1.txt','utf-8')

# 对所有的文本处理 使用tfidf #仅仅使用部分数据
def all_blog_extract_TFIDF():
    start_time = datetime.datetime.now()

    # 分词 提取关键词
    jieba.load_userdict('../data/train_blog_themes_space.txt')
    all_blogs_ID = []
    all_blog_content_list = []
    with open('1_500000.txt') as f:
        for line in f:
            line_list = line.strip().split('\001')
            ID = line_list[0]
            content = ' '.join(line_list[1:])
            all_blogs_ID.append(ID)
            all_blog_content_list.append(jieba.lcut(content))

    print "Finished loading 1_500000.txt!"

    with open('2_500000.txt') as f:
        for line in f:
            line_list = line.strip().split('\001')
            ID = line_list[0]
            content = ' '.join(line_list[1:])
            all_blogs_ID.append(ID)
            all_blog_content_list.append(jieba.lcut(content))

    print "Finished loading 2_500000.txt!"

    corpus = [' '.join(item) for item in all_blog_content_list]

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

    word = vectorizer.get_feature_names()  # 所有文本的关键字
    weight = tfidf.toarray()  # 对应的tfidf矩阵
    print len(word)
    print weight, len(weight), len(weight[0])
    print np.max(np.array(weight), axis=1)
    word_list = []
    for i in range(len(weight)):
        temp_array = weight[i]
        temp_tuple_list = []
        temp_word_list = []
        for rank, word_item in enumerate(temp_array):
            temp_tuple_list.append(tuple([word_item, rank]))
        temp_tuple_list = sorted(temp_tuple_list, key=lambda x: x[0], reverse=True)
        for word_no in range(20):
            temp_word_list.append(word[temp_tuple_list[word_no][1]])

        word_list.append(temp_word_list)

    with open('all_result_file.txt', 'wb') as f:
        for i in range(len(word_list)):
            print word_list[i]
            blog_result = all_blogs_ID[i] + '###' + '/'.join(word_list[i]) + '\n'
            print blog_result
            f.write(blog_result.encode('utf-8'))

    end_time = datetime.datetime.now()
    print 'Total time is:',end_time - start_time

    print "END..."

# all_blog_extract_TFIDF()

# 根据已经使用textrank提取关键词之后再使用TF_IDF提取
def gen_all_tfidf_themes_using_textrank_result():
    all_text_rank_word = {}
    with open(text_rank_result,'rb') as f:
        all_text_rank_word = pickle.load(f)

    print "Textrank 分词结果加载完毕！"
    # 使用tfidf提取主题关键词

    train_blog_id = []
    valid_blog_id = []
    with open(train_blog_file,'rb') as f:
        for line in f:
            id = line.strip().split('\001')[0]
            if len(id) == 11:
                id = id[3:]
            train_blog_id.append(id)
    with open(valid_blog_file,'rb') as f:
        for line in f:
            id = line.strip().split('\001')[0]
            if len(id) == 11:
                id = id[3:]
            valid_blog_id.append(id)
    ID_list = []
    # ID_list.extend(train_blog_id)
    # ID_list.extend(valid_blog_id)
    train_blog_id = set(train_blog_id)
    valid_blog_id = set(valid_blog_id)

    word_list = []
    # ID_list = all_text_rank_word.keys()

    # 添加valid、train的ID
    for id in all_text_rank_word.keys():
        if id in train_blog_id or id in valid_blog_id:
            # print id,all_text_rank_word[id]
            temp_list = all_text_rank_word[id].split('\001')
            ID_list.append(id)
            if len(temp_list) >= 2:
                # title_list = temp_list[-2].decode('utf-8').split('/')
                # content_list = temp_list[-1].decode('utf-8').split('/')
                title_list = unicode(temp_list[-2], errors='ignore').split('/')
                content_list = unicode(temp_list[-1], errors='ignore').split('/')
                title_list.extend(content_list)
            elif len(temp_list) == 1:
                title_list = unicode(temp_list[0], errors='ignore').split('/')
                # title_list = temp_list[0].decode('utf-8').split('/')
            # print len(title_list)
            word_list.append(title_list)


    for num,id in enumerate(all_text_rank_word.keys(),1):
        if num % 500 == 0:
            print "Processing lineno:",num

        if num >= 20000:
            break
        if len(id) != 8:
            continue
        else:
            # if num > 298500:
            #     print all_text_rank_word[id]
            ID_list.append(id)
            temp_list = all_text_rank_word[id].split('\001')
            if len(temp_list) >= 2:
                # title_list = temp_list[-2].decode('utf-8').split('/')
                # content_list = temp_list[-1].decode('utf-8').split('/')
                title_list = unicode(temp_list[-2],errors='ignore').split('/')
                content_list = unicode(temp_list[-1],errors='ignore').split('/')
                title_list.extend(content_list)
            elif len(temp_list) == 1:
                title_list = unicode(temp_list[0],errors='ignore').split('/')
                # title_list = temp_list[0].decode('utf-8').split('/')
            # print len(title_list)
            word_list.append(title_list)

        # break

    print "分词结果已经完成！"
    print "开始使用TF-IDF进行分词..."

    start_time = datetime.datetime.now()

    corpus = [' '.join(item) for item in word_list]

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

    word = vectorizer.get_feature_names()  # 所有文本的关键字
    weight = tfidf.toarray()  # 对应的tfidf矩阵
    print len(word)
    print weight, len(weight), len(weight[0])
    print np.max(np.array(weight), axis=1)
    word_list = []
    for i in range(len(weight)):
        temp_array = weight[i]
        temp_tuple_list = []
        temp_word_list = []
        for rank, word_item in enumerate(temp_array):
            temp_tuple_list.append(tuple([word_item, rank]))
        temp_tuple_list = sorted(temp_tuple_list, key=lambda x: x[0], reverse=True)
        for word_no in range(20):
            temp_word_list.append(word[temp_tuple_list[word_no][1]])

        word_list.append(temp_word_list)

    with open('all_result_file.txt', 'wb') as f:
        for i in range(len(word_list)):
            # print word_list[i]
            blog_result = ID_list[i] + '\001' + '/'.join(word_list[i]) + '\n'
            # print blog_result
            f.write(blog_result.encode('utf-8'))

    end_time = datetime.datetime.now()
    print 'Total time is:', end_time - start_time

    print "END..."

gen_all_tfidf_themes_using_textrank_result()

# 将train和valid数据 再加上一些数据构成100000个博客 使用tfidf提取关键词
def TF_IDF_100000_blogs():
    with open(text_rank_result,'rb') as f:
        all_themes = pickle.load(f)

    with open(train_blog_file,'rb') as f:
        train_blog_id = [line.strip()[:8] for line in f]
    with open(valid_blog_file,'rb') as f:
        valid_blog_id = [line.strip()[:8] for line in f]


# 计算得分
def get_score():
    train_result = {}
    train_origin = {}
    with open('all_result_file.txt','rb') as f:
        for line in f:
            ID,content = line.strip().split('\001')
            # print ID
            if len(ID) == 11:
                ID = ID[3:]
            train_result[ID] = content.split('/')[:3]

    with open(train_blog_file,'rb') as f:
        for line in f:
            # print line
            line_list = line.strip().split('\001')
            ID = line_list[0]
            if len(line_list[0]) == 11:
                ID = line_list[0][3:]
            # print ID
            train_origin[ID] = line_list[1:]

    num = 0
    for key in train_origin.keys():
        # print key,len(key)
        num += len(set(train_origin[key]) & set(train_result[key]))

    print 'total num:',3*len(train_origin.keys())
    print 'points:',float(num)/(3*len(train_origin.keys()))


get_score()




