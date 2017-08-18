# -*- coding:utf-8 -*-
# 日期：2017/7/20 时间：下午7:45
# Author:Jaylin
import jieba
import numpy as np
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

train_data_file = '../data/blogtheme_4zh_train.txt'
valid_data_file = '../data/blogtheme_4zh_valid.txt'
train_result_file = '../data/train_TFIDF_result.txt'
train_result_file_1 = '../data/train_TFIDF_result_1.txt'
all_result_file = '../data/all_TFIDF_result.txt'
valid_result_file = '../data/valid_TFIDF_result.txt'
train_answer_file = unicode('../data/SMP/SMPCUP2017任务1训练集/SMPCUP2017_TrainingData_Task1.txt','utf-8')
valid_id_file = unicode('../data/SMP/SMPCUP2017任务1训练集/SMPCUP2017_TrainingData_Task1.txt','utf-8')
train_blog_themes_space = '../data/train_blog_themes_space.txt'
# 生成训练博客的主题空间
def gen_train_blog_themes_space():
    all_themes = {}
    with open(train_answer_file,'rb') as f:
        for line in f:
            line_list = line.strip().split('\001')
            for word in line_list[1:]:
                if word in all_themes.keys():
                    all_themes[word] += 1
                else:
                    all_themes[word] = 1
    all_themes = sorted(all_themes.iteritems(),key= lambda x:x[1],reverse=True)
    with open(train_blog_themes_space,'wb') as f:
        for i in range(len(all_themes)):
            f.write(all_themes[i][0]+' '+str(all_themes[i][1])+'\n')

# gen_train_blog_themes_space()


# 使用tf-idf重新生成关键词 使用题目和内容 词频不考虑
def tf_idf_extract_1():
    id_list = []
    corpus = []
    with open(train_data_file,'rb') as f:
        for line in f:
            id,titel_tags,content_tags = line.strip().split('\001')
            # print id
            id_list.append(id)
            titel_tags = titel_tags.split('/')
            content_tags = content_tags.split('/')
            corpus.append(' '.join(titel_tags)+' '+' '.join(content_tags))

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

    word = vectorizer.get_feature_names()  # 所有文本的关键字
    weight = tfidf.toarray()  # 对应的tfidf矩阵
    print len(word)
    print weight,len(weight),len(weight[0])
    print np.max(np.array(weight),axis=1)
    word_list = []
    for i in range(len(weight)):
        temp_array = weight[i]
        temp_tuple_list = []
        temp_word_list = []
        for rank,word_item in enumerate(temp_array):
            temp_tuple_list.append(tuple([word_item,rank]))
        temp_tuple_list = sorted(temp_tuple_list,key=lambda x:x[0],reverse=True)
        for word_no in range(20):
            temp_word_list.append(word[temp_tuple_list[word_no][1]])

        word_list.append(temp_word_list)

    with open(train_result_file, 'wb') as f:
        for i in range(len(word_list)):
            print word_list[i]
            blog_result = id_list[i]+'###'+'/'.join(word_list[i])+'\n'
            print blog_result
            f.write(blog_result.encode('utf-8'))

# tf_idf_extract_1()  # points: 0.423026744945

# 计算得分
def get_score():
    train_result = {}
    train_origin = {}
    with open(train_result_file,'rb') as f:
        for line in f:
            ID,content = line.strip().split('###')
            # print ID
            if len(ID) == 11:
                ID = ID[3:]
            train_result[ID] = content.split('/')[:3]

    with open(train_answer_file,'rb') as f:
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
        # print key
        num += len(set(train_origin[key]) & set(train_result[key]))

    print 'total num:',3*len(train_origin.keys())
    print 'points:',float(num)/(3*len(train_origin.keys()))

# get_score()


# 加入博客内容的词频 20...1 4...1
def tf_idf_extract_2():
    id_list = []
    corpus = []
    with open(train_data_file,'rb') as f:
        for line in f:
            id,title_tags,content_tags = line.strip().split('\001')
            # print id
            id_list.append(id)
            title_tags = title_tags.split('/')
            title_tags_copy = list(title_tags)
            for item in title_tags_copy:
                print item
                title_tags.extend([item] * 3)
            print '-'*30
            content_tags = content_tags.split('/')
            # for rank,item in enumerate(content_tags,1):
            #     content_tags.extend([item]*(20-rank))
            for rank, item in enumerate(content_tags, 1):
                if rank < 5:
                    content_tags.extend([item] * 3)
                elif rank < 10:
                    content_tags.extend([item] * 2)
                elif rank < 15:
                    content_tags.extend([item] * 1)
            corpus.append(' '.join(title_tags)+' '+' '.join(content_tags))

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

    word = vectorizer.get_feature_names()  # 所有文本的关键字
    weight = tfidf.toarray()  # 对应的tfidf矩阵
    print len(word)
    print weight,len(weight),len(weight[0])
    print np.max(np.array(weight),axis=1)
    word_list = []
    for i in range(len(weight)):
        temp_array = weight[i]
        temp_tuple_list = []
        temp_word_list = []
        for rank,word_item in enumerate(temp_array):
            temp_tuple_list.append(tuple([word_item,rank]))
        temp_tuple_list = sorted(temp_tuple_list,key=lambda x:x[0],reverse=True)
        for word_no in range(20):
            temp_word_list.append(word[temp_tuple_list[word_no][1]])

        word_list.append(temp_word_list)

    with open(train_result_file, 'wb') as f:
        for i in range(len(word_list)):
            print word_list[i]
            blog_result = id_list[i]+'###'+'/'.join(word_list[i])+'\n'
            print blog_result
            f.write(blog_result.encode('utf-8'))

# tf_idf_extract_2()
# get_score()
# 根据内容或题目增加改变词频
# 19...0 points: 0.441291585127
# 3...1  points: 0.450750163079  title 2:0.492824527071 3:0.50815394651 4:0.503261578604
# 只增加题目的词频 得分较低 两个结合效果比较好

# 增加train_space的权重
def add_theme_space():
    theme_space = {}
    with open(train_blog_themes_space,'rb') as f:
        for line in f:
            line_list = line.strip().split(' ')
            word = ' '.join(line_list[:-1])
            theme_space[word] = line_list[-1]
    all_data = {}
    with open(train_result_file,'rb') as f:
        for line in f:
            id,words = line.strip().split('###')
            words = words.split('/')
            score = range(len(words),0,-1)
            # print score
            for i in range(len(words)):
                if words[i] in theme_space.keys():
                    score[i] *= float(theme_space[words[i]])

            # print score
            index_list = []
            for i in range(len(score)):
                index_list.append(tuple([score[i],i]))
            # print index_list
            index_list = sorted(index_list,key= lambda x:x[0],reverse=True)
            # print index_list
            word_list = []
            for i in range(len(index_list)):
                word_list.append(words[index_list[i][1]])
            all_data[id] = word_list
    with open(train_result_file_1, 'wb') as f:
        for key in all_data.keys():
            blog_result = key+'###'+'/'.join(all_data[key])+'\n'
            # print blog_result
            f.write(blog_result)

# add_theme_space()
# get_score()

# 生成valid结果
def gen_valid_tfidf_result():
    valid_id_list = []
    with open('../data/final.txt') as f:
        for lineno,line in enumerate(f):
            if lineno < 2:
                continue
            if line[0] == 'D':
                valid_id_list.append(line[:8])
            else:
                break

    id_list = []
    corpus = []
    with open(valid_data_file, 'rb') as f:
        for line in f:
            id, title_tags, content_tags = line.strip().split('\001')
            # print id
            id_list.append(id)
            title_tags = title_tags.split('/')
            title_tags_copy = list(title_tags)
            for item in title_tags_copy:
                print item
                title_tags.extend([item] * 3)
            print '-' * 30
            content_tags = content_tags.split('/')
            # for rank,item in enumerate(content_tags,1):
            #     content_tags.extend([item]*(20-rank))
            for rank, item in enumerate(content_tags, 1):
                if rank < 5:
                    content_tags.extend([item] * 3)
                elif rank < 10:
                    content_tags.extend([item] * 2)
                elif rank < 15:
                    content_tags.extend([item] * 1)
            corpus.append(' '.join(title_tags) + ' ' + ' '.join(content_tags))

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

    word = vectorizer.get_feature_names()  # 所有文本的关键字
    weight = tfidf.toarray()  # 对应的tfidf矩阵
    print len(word)
    print weight, len(weight), len(weight[0])
    print np.max(np.array(weight), axis=1)
    word_result = {}
    for i in range(len(weight)):
        temp_array = weight[i]
        temp_tuple_list = []
        temp_word_list = []
        for rank, word_item in enumerate(temp_array):
            temp_tuple_list.append(tuple([word_item, rank]))
        temp_tuple_list = sorted(temp_tuple_list, key=lambda x: x[0], reverse=True)
        for word_no in range(20):
            temp_word_list.append(word[temp_tuple_list[word_no][1]])

        word_result[id_list[i]] = temp_word_list

    with open(valid_result_file, 'wb') as f:
        for id in valid_id_list:
            blog_result = id+','+','.join(word_result[id][:3])+'\n'
            print blog_result
            f.write(blog_result.encode('utf-8'))

# gen_valid_tfidf_result()

# 根据TF_IDF对已经去掉停用词的训练文本提取关键词  points: 0.418134377038
def train_blog_extract_TFIDF():
    train_themes = {}
    with open(train_answer_file,'rb') as f:
        for line in f:
            line_list = line.strip().split('\001')
            ID = line_list[0][3:] if len(line_list[0]) == 11 else line_list[0]
            train_themes[ID] = line_list[1:]

    # 分词 提取关键词
    jieba.load_userdict('../data/train_blog_themes_space.txt')
    train_blogs_ID = []
    train_blog_content_list = []
    train_IDs = set(train_themes.keys())
    with open('../Desktop/1_500000.txt') as f:
        for line in f:
            line_list = line.strip().split('\001')
            ID = line_list[0]
            content = ' '.join(line_list[1:])
            if ID in train_IDs:
                train_blogs_ID.append(ID)
                train_blog_content_list.append(jieba.lcut(content))

    with open('../Desktop/2_500000.txt') as f:
        for line in f:
            line_list = line.strip().split('\001')
            ID = line_list[0]
            content = ' '.join(line_list[1:])
            if ID in train_IDs:
                train_blogs_ID.append(ID)
                train_blog_content_list.append(jieba.lcut(content))

    corpus = [' '.join(item) for item in train_blog_content_list]

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

    with open(train_result_file, 'wb') as f:
        for i in range(len(word_list)):
            print word_list[i]
            blog_result = train_blogs_ID[i] + '###' + '/'.join(word_list[i]) + '\n'
            print blog_result
            f.write(blog_result.encode('utf-8'))

# train_blog_extract_TFIDF()

# 对所有的文本处理 使用tfidf
def all_blog_extract_TFIDF():

    # 分词 提取关键词
    jieba.load_userdict('../data/train_blog_themes_space.txt')
    all_blogs_ID = []
    all_blog_content_list = []
    with open('../Desktop/1_500000.txt') as f:
        for line in f:
            line_list = line.strip().split('\001')
            ID = line_list[0]
            content = ' '.join(line_list[1:])
            all_blogs_ID.append(ID)
            all_blog_content_list.append(jieba.lcut(content))

    with open('../Desktop/2_500000.txt') as f:
        for line in f:
            line_list = line.strip().split('\001')
            ID = line_list[0]
            content = ' '.join(line_list[1:])
            all_blogs_ID.append(ID)
            all_blog_content_list.append(jieba.lcut(content))

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

    with open(all_result_file, 'wb') as f:
        for i in range(len(word_list)):
            print word_list[i]
            blog_result = all_blogs_ID[i] + '###' + '/'.join(word_list[i]) + '\n'
            print blog_result
            f.write(blog_result.encode('utf-8'))

    print "END..."

all_blog_extract_TFIDF()


















