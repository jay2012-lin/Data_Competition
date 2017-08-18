# -*-coding:UTF-8-*-
# Author:jaylin
# File:Task1_Gen_Clear_blog.py
# Time:2017/7/11 11:01
import numpy as np
import jieba
import jieba.posseg as poss
from jieba.analyse import TextRank
# from textrank4zh import TextRank4Keyword, TextRank4Sentence

word_list_file = 'test/delete_from_top_new.txt'
blog_file = unicode('E:\数据挖掘\SMP\SMPCUP2017数据集\\1_BlogContent.txt','utf-8')
train_blog_theme = unicode('E:\数据挖掘\SMP\SMPCUP2017任务1训练集\SMPCUP2017_TrainingData_Task1.txt','utf-8')
original_themes = unicode("E:\数据挖掘\SMPCUP2017\\analysis_data\original_blog_theme.txt", 'utf-8')
test_blog = '人民币金额大写人民币金额大写/*在与财务相关的应用中，经常会用到人民币金额的大写，比如发票的打印程序。本题的任务是：从键盘输入一个十亿以内的正整数（int类型），把它转换为人民币金额大写（不考虑用户输入错误的情况）。比如，用户输入：35201，程序输出：叁万伍仟贰佰零壹用户输入：30201，程序输出：叁万零贰佰零壹用户输入：30001，程序输出：叁万零壹用户输入：31000，程序输出：叁万壹仟用户输入：120023201，程序输出：壹亿贰仟零贰万叁仟贰佰零壹用户输入：120020001，程序输出：壹亿贰仟零贰万零壹用户输入：100000001，程序输出：壹亿零壹可以看到，在万后满千位，则不加零，否则要补零，但不要出现类“零零”的情况。在亿后满千万位，则不加零，否则要补零，但整个“万档”没有数字时，“万”字省去。'

def gen_clear_blog():
    word_list = []
    with open(word_list_file,'rb') as f:
        for line in f:
            line = line.strip()
            # print line,len(line)
            if line != '':
                word_list.append(line.decode('utf-8'))
    word_set = set(word_list)
    # for i in word_list:
    #     print i,len(i)
    # print len(word_list)
    # print len(list(word_set))

    jieba.load_userdict(word_list_file)
    cut_list = jieba.lcut(test_blog)
    # print len(cut_list)


    result_list = []
    for item in cut_list:
        # print item,len(item)
        if item in word_set:
            result_list.append(item)

    # print result_list
    # print len(result_list)
    #  print len(word_list),word_list[1]

def gen_user_words():
    word_list = []
    with open(word_list_file, 'rb') as f:
        for line in f:
            line = line.strip()
            # print line,len(line)
            if line != '':
                word_list.append(line.decode('utf-8'))
    word_set = set(word_list)
    word_list = list(word_set)
    with open(word_list_file[:-4]+'_.txt', 'wb') as f:
        for item in word_list:
            f.write(item.encode('utf-8')+'\n')

# gen_user_words()


# gen_clear_blog()
# 过滤所有的博客
def clear_all_blog(start_index=1,num_per=500000):
    def filter(word_set,cut_list):
        result_list = []
        for item in cut_list:
            # print item,len(item)
            if item in word_set:
                result_list.append(item.encode('utf-8'))
        return result_list


    word_list = []
    with open(word_list_file, 'rb') as f:
        for line in f:
            line = line.strip()
            # print line,len(line)
            if line != '':
                word_list.append(line.decode('utf-8'))
    word_set = set(word_list)
    out_file = str(start_index)+'_'+str(num_per)+'.txt'

    jieba.load_userdict(word_list_file)

    with open(blog_file,'rb') as f,open(out_file,'wb') as fw:
        for lineno,line in enumerate(f,1):
            if lineno <= (start_index-1)*num_per:
                continue
            if lineno > start_index*num_per:
                break
            if lineno % 500 == 0:
                print "processing lineno:",lineno
            lineID,line_title,line_content = line.strip().split('\001')
            title_result = filter(word_set,jieba.lcut(line_title))
            # print len(jieba.lcut(line_content))
            content_result = filter(word_set,jieba.lcut(line_content))
            # print len(content_result)
            fw.write(lineID+'\001'+''.join(title_result)+'\001'+''.join(content_result)+'\n')

    print "%d:%d is Finished!!!"%((start_index-1)*num_per+1,start_index*num_per)

# clear_all_blog(start_index=2)
filtered_blog = 'C:\Users\JayLin\Desktop\\1_500000.txt'
# 使用jieba的textrank主题词提取
def gen_blog_theme(start_index=1,num_per=500000):
    tokenize = jieba.Tokenizer(unicode("E:\数据挖掘\SMPCUP2017\\analysis_data\original_blog_theme.txt", 'utf-8'))
    post_tokenize = poss.POSTokenizer(tokenize)
    textrank = TextRank()
    textrank.tokenizer = post_tokenize

    blog_theme_file = 'blogtheme'+str(start_index)+'_'+str(num_per)+'.txt'
    with open(filtered_blog,'rb') as f,open(blog_theme_file,'wb') as fw:
        for lineno,line in enumerate(f,1):
            if lineno <= (start_index-1)*num_per:
                continue
            if lineno > start_index*num_per:
                break
            if lineno % 1== 0:
                print "Processing loneno:",lineno
            ID,title,content = line.strip().split('\001')
            title_tags = textrank.textrank(title, withWeight=False, withFlag=False)
            tags = textrank.textrank(content, withWeight=False, withFlag=False)
            # title_tags = [item.encode('utf-8') for item in title_tags]
            # tags = [item.encode('utf-8') for item in tags]

            fw.write(ID+'\001'+'/'.join(title_tags).encode('utf-8')+'\001'+'/'.join(tags).encode('utf-8')+'\n')

# gen_blog_theme(start_index=1,num_per=20)
def gen_blog_them_by_4zh(start_index=1,num_per=500000):
    tr4w = TextRank4Keyword()
    blog_theme_file = 'blogtheme_4zh_' + str(start_index) + '_' + str(num_per) + '.txt'
    with open(filtered_blog, 'rb') as f, open(blog_theme_file, 'wb') as fw:
        for lineno, line in enumerate(f, 1):
            if lineno <= (start_index - 1) * num_per:
                continue
            if lineno > start_index * num_per:
                break
            if lineno % 50 == 0:
                print "Processing loneno:", lineno
            ID, title, content = line.strip().split('\001')

            tr4w.analyze(text=title, lower=True, window=4)
            title_tags = []
            for item in tr4w.get_keywords(20, word_min_len=2):
                title_tags.append(item.word)
            tr4w.analyze(text=content, lower=True, window=4)
            tags = []
            for item in tr4w.get_keywords(20, word_min_len=2):
                tags.append(item.word)

            # title_tags = textrank.textrank(title, withWeight=False, withFlag=False)
            # tags = textrank.textrank(content, withWeight=False, withFlag=False)
            # title_tags = [item.encode('utf-8') for item in title_tags]
            # tags = [item.encode('utf-8') for item in tags]

            fw.write(ID + '\001' + '/'.join(title_tags).encode('utf-8') + '\001' + '/'.join(tags).encode('utf-8') + '\n')

    print "Finished!!!"

# gen_blog_them_by_4zh(start_index=2)
# gen_blog_them_by_4zh(start_index=5,num_per=100000)
# 通过行号读取
def gen_blog_them_by_4zh_no(start_lineno=1,num_per=500000):
    existed_id_list = []
    with open('1_exist.txt','rb') as f:
        for line in f:
            existed_id_list.append(line.strip())
    existed_id_set = set(existed_id_list)

    tr4w = TextRank4Keyword()
    blog_theme_file = 'blogtheme_4zh_' + str(start_lineno) + '_' + str(num_per) + '.txt'
    with open(filtered_blog, 'rb') as f, open(blog_theme_file, 'wb') as fw:
        for lineno, line in enumerate(f, 1):
            if lineno <= start_lineno:
                continue
            if lineno > start_lineno + num_per:
                break
            if lineno % 50 == 0:
                print "Processing loneno:", lineno
            ID, title, content = line.strip().split('\001')
            if ID in existed_id_set:
                continue

            tr4w.analyze(text=title, lower=True, window=4)
            title_tags = []
            for item in tr4w.get_keywords(20, word_min_len=2):
                title_tags.append(item.word)
            tr4w.analyze(text=content, lower=True, window=4)
            tags = []
            for item in tr4w.get_keywords(20, word_min_len=2):
                tags.append(item.word)

            # title_tags = textrank.textrank(title, withWeight=False, withFlag=False)
            # tags = textrank.textrank(content, withWeight=False, withFlag=False)
            # title_tags = [item.encode('utf-8') for item in title_tags]
            # tags = [item.encode('utf-8') for item in tags]

            fw.write(ID + '\001' + '/'.join(title_tags).encode('utf-8') + '\001' + '/'.join(tags).encode('utf-8') + '\n')

    print "Finished!!!"

# gen_blog_them_by_4zh_no(start_lineno=300000,num_per=200000)


def gen_blog_them_by_4zh_use_trainblogID(start_index=1,num_per=500000):
    user_ID = []
    with open('../analysis_data/result_data/task1/task1_train_delete_stop_word.txt','rb') as f:
        for item in f:
            user_ID.append(item.split('	')[0])
    print user_ID
    user_ID = set(user_ID)


    tr4w = TextRank4Keyword()
    blog_theme_file = 'blogtheme_4zh_train_' + str(2) + '_' + str(num_per) + '.txt'
    iter_num = 0
    with open('C:\Users\JayLin\Desktop\\2_500000.txt', 'rb') as f, open(blog_theme_file, 'wb') as fw:
        for lineno, line in enumerate(f, 1):
            # if iter_num > 50:
            #     break
            if lineno <= (start_index - 1) * num_per:
                continue
            if lineno > start_index * num_per:
                break
            if lineno % 1000 == 0:
                print "Processing loneno:", lineno
            ID, title, content = line.strip().split('\001')
            if ID not in user_ID:
                continue

            iter_num += 1

            print '训练数据处理：',iter_num

            tr4w.analyze(text=title, lower=True, window=4)
            title_tags = []
            for item in tr4w.get_keywords(20, word_min_len=2):
                title_tags.append(item.word)
            tr4w.analyze(text=content, lower=True, window=4)
            tags = []
            for item in tr4w.get_keywords(20, word_min_len=2):
                tags.append(item.word)

            # title_tags = textrank.textrank(title, withWeight=False, withFlag=False)
            # tags = textrank.textrank(content, withWeight=False, withFlag=False)
            # title_tags = [item.encode('utf-8') for item in title_tags]
            # tags = [item.encode('utf-8') for item in tags]

            fw.write(ID + '\001' + '/'.join(title_tags).encode('utf-8') + '\001' + '/'.join(tags).encode('utf-8') + '\n')

    print "Finished!!!"

# gen_blog_them_by_4zh_use_trainblogID()
candicate_file = 'blogtheme_4zh_train.txt'
theme_result_file = 'blogtheme_result.txt'
# 从已经生成的候选词里面按照一定的规律排序候选词  拟合结果
# 做法是，对内容的主题结果按照长度排序 根据是否在题目主题词中出现和theme_space中出现增加权重
def select_result_theme(title2content=2,content2existed=2):
    all_train_themes = {}
    def f_cmp(a,b):
        return len(a)-len(b)

    train_blogtheme_dic = {}
    with open(train_blog_theme,'rb') as f:
        for line in f:
            list_ = line.strip().split('\001')
            ID = list_[0]
            if len(ID) == 11:
                ID = ID[3:]
            blog_theme = list_[1:]
            train_blogtheme_dic[ID] = blog_theme

    train_blogthemes = []
    with open(original_themes, 'rb') as f:
        for line in f:
            list_ = line.strip().split(' ')
            train_blogthemes.append(''.join(list_[:-2]))
            # print ''.join(list_[:-2])

    with open(candicate_file,'rb') as f:
        for line in f:
            ID,title_tags,tags = line.strip().split('\001')
            title_tags = title_tags.split('/')
            tags = tags.split('/')
            tags = sorted(tags,cmp=f_cmp,reverse=True)
            tags_dic = {}
            tags_num = len(tags)
            for i in range(tags_num,0,-1):
                # print i,tags[i-1]
                tags_dic[tags[i-1]] = tags_num+1-i
            # tags_no = zip(tags,range(len(tags)))
            # print tags_,tags
            # print tags_dic
            for item in title_tags:
                if item in tags:
                    # print item
                    tags_dic[item] = tags_dic[item]*title2content
            # print tags_dic

            for item in tags:
                if item in set(train_blogthemes):
                    # print item
                    tags_dic[item] = tags_dic[item] * content2existed
            # print tags_dic
            for item in title_tags:
                if len(item)>=6 and item not in tags:
                    if item in set(train_blog_theme):
                        tags_dic[item] = tags_num+1


            tags_result = sorted(tags_dic.iteritems(),key=lambda x:x[1],reverse=True)
            tags_result = [item[0] for item in tags_result]
            # print tags_result
            all_train_themes[ID] = tags_result

            # print tags_result
            # break
    # 计算得分
    num_rights = 0
    num_blogs = 0
    for key in all_train_themes.keys():
        predicted_themes = set(all_train_themes[key][:3])
        existed_themes = set(train_blogtheme_dic[key])
        num_rights += len(predicted_themes & existed_themes)
        num_blogs += 1

    print "匹配的相同主题数量为：",num_rights
    point = float(num_rights)/(num_blogs*3)
    print point


    with open(theme_result_file,'wb') as f:
        for key in all_train_themes.keys():
            f.write(key+','+','.join(all_train_themes[key])+'\n')

    return point,title2content,content2existed


# select_result_theme(title2content=1.0,content2existed=10)
def train_select_model():
    result = []
    title2content = 0.1
    while title2content <= 3.5:
        content2existed = 0.1
        while content2existed <= 3.5:
            print title2content, content2existed
            result.append(select_result_theme(title2content=title2content, content2existed=content2existed))
            content2existed += 0.1
        title2content += 0.1
    result = sorted(result, key=lambda x: x[0], reverse=True)
    print result[0]

# train_select_model()


# 预测模型
# 2 和2.3效果最好
def gen_valid_candicate(start_index=1,num_per=500000):
    user_ID = []
    with open('../analysis_data/SMPCUP2017_ValidationSet_Task1.txt','rb') as f:
        for item in f:
            user_ID.append(item.strip())
    print user_ID
    user_ID = set(user_ID)


    tr4w = TextRank4Keyword()
    blog_theme_file = 'blogtheme_4zh_valid_' + str(2) + '_' + str(num_per) + '.txt'
    iter_num = 0
    with open('C:\Users\JayLin\Desktop\\2_500000.txt', 'rb') as f, open(blog_theme_file, 'wb') as fw:
        for lineno, line in enumerate(f, 1):
            # if iter_num > 50:
            #     break
            if lineno <= (start_index - 1) * num_per:
                continue
            if lineno > start_index * num_per:
                break
            if lineno % 1000 == 0:
                print "Processing loneno:", lineno
            ID, title, content = line.strip().split('\001')
            if ID not in user_ID:
                continue

            iter_num += 1

            print 'valid数据处理：',iter_num

            tr4w.analyze(text=title, lower=True, window=4)
            title_tags = []
            for item in tr4w.get_keywords(20, word_min_len=2):
                title_tags.append(item.word)
            tr4w.analyze(text=content, lower=True, window=4)
            tags = []
            for item in tr4w.get_keywords(20, word_min_len=2):
                tags.append(item.word)

            # title_tags = textrank.textrank(title, withWeight=False, withFlag=False)
            # tags = textrank.textrank(content, withWeight=False, withFlag=False)
            # title_tags = [item.encode('utf-8') for item in title_tags]
            # tags = [item.encode('utf-8') for item in tags]

            fw.write(ID + '\001' + '/'.join(title_tags).encode('utf-8') + '\001' + '/'.join(tags).encode('utf-8') + '\n')

    print "Finished!!!"

# gen_valid_candicate()

def select_valid_result_theme(title2content=2,content2existed=2):
    user_ID = []
    with open('../analysis_data/SMPCUP2017_ValidationSet_Task1.txt', 'rb') as f:
        for item in f:
            user_ID.append(item.strip())
    print user_ID


    all_valid_themes = {}
    def f_cmp(a,b):
        return len(a)-len(b)

    train_blogthemes = []
    with open(original_themes, 'rb') as f:
        for line in f:
            list_ = line.strip().split(' ')
            train_blogthemes.append(''.join(list_[:-2]))
            # print ''.join(list_[:-2]),len(''.join(list_[:-2]))

    with open('blogtheme_4zh_valid.txt','rb') as f:
        for line in f:
            ID,title_tags,tags = line.strip().split('\001')
            if len(ID) == 1:
                ID = ID[3:]
            title_tags = title_tags.split('/')
            tags = tags.split('/')
            tags = sorted(tags,cmp=f_cmp,reverse=True)
            tags_dic = {}
            tags_num = len(tags)
            for i in range(tags_num,0,-1):
                tags_dic[tags[i-1]] = tags_num+1-i
            for item in title_tags:
                if item in tags:
                    # print item
                    tags_dic[item] = tags_dic[item]*title2content
            # print tags_dic

            for item in tags:
                # print len(item),item
                if item in set(train_blogthemes):
                    # print item
                    tags_dic[item] = tags_dic[item] * content2existed
            # print tags_dic
            for item in title_tags:
                # print len(item),item
                if len(item)>6 and item not in tags:
                    # print item,len(item)
                    if item in set(train_blogthemes):
                        # print item
                        tags_dic[item] = 2*tags_num+1

            # for item in title

            tags_result = sorted(tags_dic.iteritems(),key=lambda x:x[1],reverse=True)
            tags_result = [item[0] for item in tags_result]
            # print tags_result
            all_valid_themes[ID] = tags_result

    with open('valid_themes.txt','wb') as f:
        # for key in all_valid_themes.keys():
        for user_id in user_ID:
            print user_id
            f.write(user_id+','+','.join(all_valid_themes[user_id][:3])+'\n')

# 2.0 2.3 0.48076
# 2.0 2.3 0.50130
# 2.0 3.0 0.47大约
# select_valid_result_theme(title2content=2.0,content2existed=2.3)

# 对theme_space中词按照词频进行排序
# 两个参数都有用 相对的目标不同
def select_result_theme_1(length_weight=2,title2content=2.0,content2existed=1.0):
    # 已经存在主题词的文本 用于打分
    train_blogtheme_dic = {}
    with open(train_blog_theme, 'rb') as f:
        for line in f:
            list_ = line.strip().split('\001')
            ID = list_[0]
            if len(ID) == 11:
                ID = ID[3:]
            blog_theme = list_[1:]
            train_blogtheme_dic[ID] = blog_theme

    theme_space_dic = {}
    with open(original_themes,'rb') as f:
        for line in f:
            line_list = line.strip().split(' ')
            word = line_list[:-2]
            count = line_list[-2]
            # print count
            theme_space_dic[" ".join(word)] = int(count)

    theme_space_list = sorted(theme_space_dic.iteritems(),key=lambda x:x[1],reverse=True)
    theme_space_count = [item[1] for item in theme_space_list]
    # print theme_space_count
    theme_space_list = [item[0] for item in theme_space_list]
    # 归一化之后加1
    theme_space_count = np.array(theme_space_count)/float(theme_space_count[0]) + content2existed

    # 生成新的主题词词典
    theme_space_dic = {}
    for i in range(len(theme_space_list)):
        theme_space_dic[theme_space_list[i]] = theme_space_count[i]

    # print theme_space_count
    # for count in theme_space_dic.values():
    #     print count,
    # print
    # print np.array(theme_space_dic.values()).max()



    # with open("themes.txt",'wb') as f:
    #     for item in theme_space_list:
    #         f.write(item[0]+" "+str(item[1])+'\n')

    all_train_themes = {}
    with open(candicate_file,'rb') as f:
        for line in f:
            ID,title_tags,tags = line.strip().split('\001')
            title_tags = set(title_tags.split('/'))
            tags = tags.split('/')

            # tags_scores = list(np.ones(len(tags),))
            tags_scores = range(len(tags),0,-1)
            # print tags_scores
            tags_dic ={}
            for i in range(len(tags)):
                tags_dic[tags[i]] = tags_scores[i]
            # print tags_dic

            for key in tags_dic.keys():
                # print len(key.decode('utf-8'))
                # 这儿最好做个判断 长度是否大于2 表示以此为基准
                if len(key.decode('utf-8')) > 2:
                    tags_dic[key] *= len(key.decode('utf-8')) * length_weight
                if key in title_tags:
                    tags_dic[key] *= title2content
                if key in theme_space_dic.keys():
                    tags_dic[key] *= theme_space_dic[key]

            # print tags_dic.values()

            tags_sorted = sorted(tags_dic.iteritems(),key= lambda x:x[1],reverse=True)
            all_train_themes[ID] = [item[0] for item in tags_sorted]

    with open('train_result.txt','wb') as f:
        for key in all_train_themes.keys():
            f.write(key+" "+" ".join(all_train_themes[key])+'\n')



    # 计算得分
    num_rights = 0
    num_blogs = 0
    for key in all_train_themes.keys():
        predicted_themes = set(all_train_themes[key][:3])
        # print predicted_themes
        existed_themes = set(train_blogtheme_dic[key])
        # print existed_themes
        num_rights += len(predicted_themes & existed_themes)
        num_blogs += 1

    # print num_blogs


    point = float(num_rights)/(num_blogs*3)
    # print "匹配的相同主题数量为：", num_rights
    # print point
    #
    # print "END..."
    return point

# select_result_theme_1(length_weight=3,title2content=7,content2existed=8)

def train_weights():
    result_score = []
    length_weight = 2.5
    while length_weight < 3.5:
        title2content = 6.5
        while title2content < 7.5:
            content2existed = 7.5
            while content2existed < 8.5:
                result = select_result_theme_1(length_weight=length_weight,title2content=title2content,content2existed=content2existed)
                print length_weight,title2content,content2existed,result
                result_score.append([length_weight,title2content,content2existed,result])
                content2existed += 0.1
            title2content += 0.1
        length_weight += 0.1

    result_score = sorted(result_score,key= lambda x:x[3],reverse=True)
    print "最优的参数为：",result_score[0]
    # [3.0, 7.0, 8.0, 0.5466405740378343]
    print "END..."

# train_weights()

def select_valid_result_theme_1(length_weight=3,title2content=7,content2existed=8):
    user_ID = []
    with open('../analysis_data/SMPCUP2017_ValidationSet_Task1.txt', 'rb') as f:
        for item in f:
            user_ID.append(item.strip())
    print user_ID

    theme_space_dic = {}
    with open(original_themes, 'rb') as f:
        for line in f:
            line_list = line.strip().split(' ')
            word = line_list[:-2]
            count = line_list[-2]
            # print count
            theme_space_dic[" ".join(word)] = int(count)

    theme_space_list = sorted(theme_space_dic.iteritems(), key=lambda x: x[1], reverse=True)
    theme_space_count = [item[1] for item in theme_space_list]
    # print theme_space_count
    theme_space_list = [item[0] for item in theme_space_list]
    # 归一化之后加1
    theme_space_count = np.array(theme_space_count) / float(theme_space_count[0]) + content2existed

    # 生成新的主题词词典
    theme_space_dic = {}
    for i in range(len(theme_space_list)):
        theme_space_dic[theme_space_list[i]] = theme_space_count[i]

    all_valid_themes = {}
    with open('blogtheme_4zh_valid.txt', 'rb') as f:
        for line in f:
            ID, title_tags, tags = line.strip().split('\001')
            title_tags = set(title_tags.split('/'))
            tags = tags.split('/')

            tags_scores = range(len(tags), 0, -1)
            # print tags_scores
            tags_dic = {}
            for i in range(len(tags)):
                tags_dic[tags[i]] = tags_scores[i]
            # print tags_dic

            for key in tags_dic.keys():
                # print len(key.decode('utf-8'))
                # 这儿最好做个判断 长度是否大于2 表示以此为基准
                if len(key.decode('utf-8')) > 2:
                    tags_dic[key] *= len(key.decode('utf-8')) * length_weight
                if key in title_tags:
                    tags_dic[key] *= title2content
                if key in theme_space_dic.keys():
                    tags_dic[key] *= theme_space_dic[key]

            # print tags_dic.values()

            tags_sorted = sorted(tags_dic.iteritems(), key=lambda x: x[1], reverse=True)
            all_valid_themes[ID] = [item[0] for item in tags_sorted]

    with open('valid_themes.txt','wb') as f:
        # for key in all_valid_themes.keys():
        for user_id in user_ID:
            print user_id
            f.write(user_id+','+','.join(all_valid_themes[user_id][:3])+'\n')

# select_valid_result_theme_1(length_weight=3.5,title2content=1.5,content2existed=1.5)

# 分别计算题目、内容、题目加内容的得分
def get_score():
    title_tags = {}
    content_tags ={}
    all_tags = {}
    tags_space = set()
    with open(candicate_file ,'rb') as f:
        for line in f:
            ID,title,content = line.strip().split('\001')
            if len(ID) == 11:
                ID = ID[3:]
            title_tags_set = set(title.split('/'))
            content_tags_set = set(content.split('/'))
            all_tags_set = title_tags_set | content_tags_set
            title_tags[ID] = title_tags_set
            content_tags[ID] = content_tags_set
            all_tags[ID] = all_tags_set
            tags_space = tags_space | all_tags_set

    title_tags_num = 0
    content_tags_num = 0
    all_tags_num = 0
    tags_space_train = set()
    with open(train_blog_theme,'rb') as f:
        for line in f:
            line_list = line.strip().split('\001')
            ID = line_list[0]
            if len(ID) == 11:
                ID = ID[3:]
            tags = set(line_list[1:])
            tags_space_train = tags_space_train | tags
            title_tags_num += len(tags & title_tags[ID])
            content_tags_num += len(tags & content_tags[ID])
            all_tags_num += len(tags & all_tags[ID])

    num = len(all_tags.keys())
    print "总共的标签个数为：",5 * num
    print "标题匹配个数为：",title_tags_num,'得分为：',float(title_tags_num)/num/5,float(title_tags_num)/num
    print "内容匹配个数为：",content_tags_num,'得分为：',float(content_tags_num)/num/5,float(content_tags_num)/num
    print "总共匹配个数为：",all_tags_num,'得分为：',float(all_tags_num)/num/5,float(all_tags_num)/num
    print len(tags_space),len(tags_space_train)  # 4137 2310
    print "主题空间的交集大小为：",len(tags_space & tags_space_train)

    # 总共的标签个数为： 5110
    # 标题匹配个数为： 1449 得分为： 0.283561643836 1.41780821918
    # 内容匹配个数为： 2964 得分为： 0.580039138943 2.90019569472
    # 总共匹配个数为： 3506 得分为： 0.686105675147 3.43052837573
    # 主题空间的交集大小为： 1897 总的为：2309
# get_score()
import pickle
all_blog_themes_file = '../tag_pickles/all_blog_themes.pickle'
def gen_all_blog_tags():
    all_themes = {}
    with open('../tag_pickles/all_blog_themes.pickle','rb') as f:
        all_themes = pickle.load(f)

    prefix = 'C:\Users\JayLin\Desktop\\'
    file_list = [
                 # 'blogtheme_4zh_300000_200000.txt'
                 prefix + 'blogtheme_4zh_1_100000.txt',prefix + 'blogtheme_4zh_2_100000.txt',prefix + 'blogtheme_4zh_3_100000.txt'
                 ]

    for file in file_list:
        print "-"*20+"Processing file:",file
        print "字典长度为：", len(all_themes.keys())
        with open(file,'rb') as f:
            for lineno,line in enumerate(f,1):
                if lineno % 500 == 0:
                    print "processing lineno:",lineno
                line_list = line.strip().split('\001')
                if len(line_list[0]) == 11:
                    id = line_list[0][3:]
                else:
                    id = line_list[0]
                all_themes[id] = '\001'.join(line_list[1:])

    print "字典长度为：",len(all_themes.keys())
    # 目前 650046 行数据  690046 700001  1000001
    with open(all_blog_themes_file,'wb') as f:
       pickle.dump(all_themes,f)

# gen_all_blog_tags()
# 检查原始两个文本是否正确分开
prefix = 'C:\Users\JayLin\Desktop\\'
def varify_1():
    all_blog_id = []
    with open(prefix+'1_500000.txt') as f:
        for lineno,line in enumerate(f,1):
            if lineno % 1000 == 0:
                print lineno
            id = line.strip().split('\001')[0]
            if len(id) == 11:
                id = id[3:]
            all_blog_id.append(id)
    print "-"*30
    with open(prefix+'2_500000.txt') as f:
        for lineno,line in enumerate(f,1):
            if lineno % 1000 == 0:
                print lineno
            id = line.strip().split('\001')[0]
            if len(id) == 11:
                id = id[3:]
            all_blog_id.append(id)


    print len(all_blog_id)
    print len(set(all_blog_id))
# varify_1()
# 验证那些数据已经处理
def varify_2():
    exist_blog = {}
    with open('../tag_pickles/all_blog_themes.pickle','rb') as f:
        exist_blog = pickle.load(f)

    exist_id = set(exist_blog.keys())
    print 'load pickle finished!!!'

    id_num_1 = 0
    with open(prefix+'1_500000.txt') as f:
        for lineno,line in enumerate(f,1):
            if lineno % 500 == 0:
                print lineno
            id = line.strip().split('\001')[0]
            if len(id) == 11:
                id = id[3:]
            if id in exist_id:
                id_num_1 += 1
    print "-"*30
    id_num_2 = 0
    with open(prefix+'2_500000.txt') as f:
        for lineno,line in enumerate(f,1):
            if lineno % 500 == 0:
                print lineno
            id = line.strip().split('\001')[0]
            if len(id) == 11:
                id = id[3:]
            if id in exist_id:
                id_num_2 += 1

    print id_num_1,id_num_2
    # 190045 460000
#varify_2()

# 验证第一组数据是否正确切分
def varify_3():
    pre_id = []
    with open(prefix+'1_500000.txt','rb') as f:
        for line in f:
            id = line.strip().split('\001')[0]
            if len(id) == 11:
                id = id[3:]
            pre_id.append(id)

    after_id = []
    with open(prefix+'blogtheme_4zh_1_500000.txt','rb') as f:
        for lineno,line in enumerate(f,1):
            if lineno % 100000 == 0:
                print len(set(pre_id) & set(after_id))
                # after_id = []
            id = line.strip().split('\001')[0]
            if len(id) == 11:
                id = id[3:]
            after_id.append(id)

    print len(pre_id),len(after_id)
    print len(set(pre_id) & set(after_id))
    with open('1_exist.txt','wb') as f:
        for id in set(pre_id) & set(after_id):
            f.write(id+'\n')
# varify_3()
# with open('../tag_pickles/all_blog_themes.pickle','rb') as f:
#     print "1"*10
#     data = pickle.load(f)
#     print len(data.keys())
#     print data['D0810948']
#     print data['D0850927']
#     print 'D0896169' in data.keys()
#     for key in data.keys():
#         if len(key) != 8:  # D09775121694 不合理的编号
#             print key,data[key]
#     print data['D0977512']

def gen_test_task1_result():
    test_id_list = []
    with open('../test/SMPCUP2017_TestSet_Task1.txt','rb') as f:
        for id in f:
            id = id.strip()
            test_id_list.append(id if len(id) == 8 else id[3:])


    print len(test_id_list),test_id_list

    with open('all_blog_themes.pickle','rb') as f:
        all_blog_themes = pickle.load(f)

    print all_blog_themes['D0000070']

    origin_test_themes = {}
    for item in test_id_list:
        origin_test_themes[item] = all_blog_themes[item]


    # print type(all_blog_themes)
    length_weight = 3
    title2content = 7
    content2existed = 8

    theme_space_dic = {}
    # with open(original_themes, 'rb') as f:
    #     for line in f:
    #         line_list = line.strip().split(' ')
    #         word = line_list[:-2]
    #         count = line_list[-2]
    #         # print count
    #         theme_space_dic[" ".join(word)] = int(count)

    train_blogs_file = '../data/train_blog_themes_space.txt'
    with open(train_blogs_file,'rb') as f:
        for line in f:
            line_list = line.strip().split(' ')
            theme_space_dic[' '.join(line_list[:-1])] = int(line_list[-1])

    theme_space_list = sorted(theme_space_dic.iteritems(), key=lambda x: x[1], reverse=True)
    theme_space_count = [item[1] for item in theme_space_list]
    # print theme_space_count
    theme_space_list = [item[0] for item in theme_space_list]
    # 归一化之后加1
    theme_space_count = np.array(theme_space_count) / float(theme_space_count[0]) + content2existed

    # 生成新的主题词词典
    theme_space_dic = {}
    for i in range(len(theme_space_list)):
        theme_space_dic[theme_space_list[i]] = theme_space_count[i]

    all_test_themes = {}
    for id in origin_test_themes.keys():
        title_tags, tags = origin_test_themes[id].strip().split('\001')
        title_tags = title_tags.split('/')
        tags = tags.split('/')

        # 删除不在标签空间中出现的词
        # print len(tags),tags
        theme_spaces = theme_space_dic.keys()
        # print theme_spaces
        # print 'er' in theme_spaces
        # break
        for item in title_tags:
            if item not in theme_spaces:
                title_tags.remove(item)
        for item in tags:
            if item not in theme_spaces:
                tags.remove(item)

        # print len(tags),tags

        tags_scores = range(len(tags), 0, -1)
        # print tags_scores
        tags_dic = {}
        for i in range(len(tags)):
            tags_dic[tags[i]] = tags_scores[i]
        # print tags_dic

        for key in tags_dic.keys():
            # print len(key.decode('utf-8'))
            # 这儿最好做个判断 长度是否大于2 表示以此为基准
            if len(key.decode('utf-8')) > 2:
                tags_dic[key] *= len(key.decode('utf-8')) * length_weight
            if key in title_tags:
                tags_dic[key] *= title2content
            if key in theme_space_dic.keys():
                tags_dic[key] *= theme_space_dic[key]

        # print tags_dic.values()

        tags_sorted = sorted(tags_dic.iteritems(), key=lambda x: x[1], reverse=True)
        all_test_themes[id] = [item[0] for item in tags_sorted]

    with open('task1_result.txt', 'wb') as f:
        for user_id in test_id_list:
            # print user_id
            f.write(user_id + ',' + ','.join(all_test_themes[user_id][:3]) + '\n')

gen_test_task1_result()
