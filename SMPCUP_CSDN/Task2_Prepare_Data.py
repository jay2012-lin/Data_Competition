# -*-coding:UTF-8-*-
# Author:jaylin
# File:Task2_Prepare_Data.py
# Time:2017/7/24 14:36
import pickle
# 使用textrank生成task2的输入数据集
all_themes_file = 'all_blog_themes.pickle'
train_user_file = unicode('../data/SMP/SMPCUP2017任务2训练集/SMPCUP2017_TrainingData_Task2.txt','utf-8')
# valid_user_file = unicode('../data/SMPCUP2017验证集/SMPCUP2017_ValidationSet_Task2.txt','utf-8')
valid_user_file = unicode('../test/SMPCUP2017_TestSet_Task2.txt','utf-8')
activity_file_profix = '../data/SMP/SMPCUP2017数据集/'

# result_profix = '../data/Task2/'
result_profix = '../data/Task2/Test/'


# 加载所有博客的主题词
def load_blog_themes():
    with open(all_themes_file,'rb') as f:
        return pickle.load(f)

def get_train_id():
    with open(train_user_file,'rb') as f:
        return [line[:8] for line in f]

def get_valid_id():
    with open(valid_user_file,'rb') as f:
        return [line[:8] for line in f]
# print get_train_id()
# print get_valid_id()

# 根据不同的类别生成不同的训练文件和预测文件
def gen_txt():
    all_themes = load_blog_themes()
    print '主题词加载完毕！'
    train_id = set(get_train_id())
    valid_id = set(get_valid_id())
    activity_list = ['Post','Browse','Comment','Vote-up','Vote-down','Favorite']
    for no,activity in enumerate(activity_list,2):
        activity_train = {}
        activity_valid = {}
        file_name = unicode(activity_file_profix+str(no)+'_'+activity+'.txt','utf-8')
        with open(file_name,'rb') as f:
            for line in f:
                line_list = line.strip().split('\001')
                id = line_list[0][3:] if len(line_list[0]) == 11 else line_list[0]
                blog_id = line_list[1]
                if id in train_id:
                    activity_train[id+':'+blog_id] = all_themes[blog_id]
                if id in valid_id:
                    activity_valid[id+':'+blog_id] = all_themes[blog_id]

        print '训练数据：',len(activity_train.keys())
        print '验证数据：',len(activity_valid.keys())

        # 写入文件
        train_file_name = result_profix + str(no) + "_trian_" + activity + '.txt'
        valid_file_name = result_profix + str(no) + "_valid_" + activity + '.txt'
        with open(train_file_name,'wb') as f:
            for item in activity_train.keys():
                f.write(item+'\001'+activity_train[item]+'\n')

        with open(valid_file_name,'wb') as f:
            for item in activity_valid.keys():
                f.write(item+'\001'+activity_valid[item]+'\n')

        print "Finished processing %s !"%activity

gen_txt()
