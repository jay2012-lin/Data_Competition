#-*- coding:UTF-8 -*-
# author:jaylin
# time:17-7-3 上午9:58
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import datetime,time,math


#数据都是一百万条
post_data_path = '../data/2_Post.txt'  # 1000000条
browse_data_path = '../data/3_Browse.txt'  # 3536444条
comment_data_path = '../data/4_Comment.txt'  # 182273
voteup_data_path = '../data/5_Vote-up.txt'  # 95668
votedown_data_path = '../data/6_Vote-down.txt'  # 9326
favorite_data_path = '../data/7_Favorite.txt'  # 104723
follow_data_path = '../data/8_Follow.txt'  # 667037
letter_data_path = '../data/9_Letter.txt'  # 46572

# 将所有的用户数据组成训练数据集
# 157427条记录
def process_useraction_data():
    # 发布博客记录
    post_data = np.loadtxt(post_data_path,dtype=str,delimiter='\001')
    post_data = pd.DataFrame(post_data,columns=('id','blogid','time'))
    post_data = post_data.groupby(post_data['id']).count()

    user_data = post_data.drop(['blogid',],axis=1)  # 不改变原DataFrame 返回新的
    user_data.rename(columns={'time':'post_num'},inplace=True)
    print "博客记录处理完毕！"

    # 浏览记录
    browse_data = np.loadtxt(browse_data_path,dtype=str,delimiter='\001')
    browse_data = pd.DataFrame(browse_data,columns=('id','blogid','time'))
    browse_data = browse_data.groupby(browse_data['id']).count()
    browse_data = browse_data.drop(['blogid', ], axis=1)  # 不改变原DataFrame 返回新的
    browse_data.rename(columns={'time': 'browse_num'}, inplace=True)  # inplace = True 表示直接在原记录上修改
    user_data = user_data.join(browse_data,how='outer')
    print "浏览记录处理完毕！"

    # 评论记录
    comment_data = np.loadtxt(comment_data_path,dtype=str,delimiter='\001')
    comment_data = pd.DataFrame(comment_data,columns=('id','blogid','time'))
    comment_data = comment_data.groupby(comment_data['id']).count()
    comment_data = comment_data.drop(['blogid'],axis=1)
    comment_data.rename(columns={'time':'comment_num'},inplace=True)
    user_data = user_data.join(comment_data,how='outer')
    print "评论记录处理完毕！"

    # 点赞记录
    voteup_data = np.loadtxt(voteup_data_path, dtype=str, delimiter='\001')
    voteup_data = pd.DataFrame(voteup_data, columns=('id', 'blogid', 'time'))
    voteup_data = voteup_data.groupby(voteup_data['id']).count()
    voteup_data = voteup_data.drop(['blogid'], axis=1)
    voteup_data.rename(columns={'time': 'voteup_num'}, inplace=True)
    user_data = user_data.join(voteup_data, how='outer')
    print "点赞记录处理完毕！"

    # 踩记录
    votedown_data = np.loadtxt(votedown_data_path, dtype=str, delimiter='\001')
    votedown_data = pd.DataFrame(votedown_data, columns=('id', 'blogid', 'time'))
    votedown_data = votedown_data.groupby(votedown_data['id']).count()
    votedown_data = votedown_data.drop(['blogid'], axis=1)
    votedown_data.rename(columns={'time': 'votedown_num'}, inplace=True)
    user_data = user_data.join(votedown_data, how='outer')
    print "踩记录处理完毕！"

    # 收藏记录
    favorite_data = np.loadtxt(favorite_data_path, dtype=str, delimiter='\001')
    favorite_data = pd.DataFrame(favorite_data, columns=('id', 'blogid', 'time'))
    favorite_data = favorite_data.groupby(favorite_data['id']).count()
    favorite_data = favorite_data.drop(['blogid'], axis=1)
    favorite_data.rename(columns={'time': 'favorite_num'}, inplace=True)
    user_data = user_data.join(favorite_data, how='outer')
    print "收藏记录处理完毕！"

    # 追随记录
    follow_data = np.loadtxt(follow_data_path,dtype=str,delimiter='\001')
    follow_data = pd.DataFrame(follow_data,columns=('follow','fan'))
    fan_data = follow_data.groupby(follow_data['fan']).count()
    fan_data.rename(columns={'follow':'follow_num'},inplace=True)
    follow_data = follow_data.groupby(follow_data['follow']).count()
    follow_data.rename(columns={'fan':'fan_num'},inplace=True)
    user_data = user_data.join(fan_data,how='outer')
    user_data = user_data.join(follow_data,how='outer')
    print "追随记录处理完毕！"

    # 私信记录
    letter_data = np.loadtxt(letter_data_path,dtype=str,delimiter='\001')
    letter_data = pd.DataFrame(letter_data,columns=('user_from','user_to','time'))
    user_from = letter_data.groupby(letter_data['user_from']).count()
    user_from = user_from.drop(['user_to'],axis=1)
    user_from.rename(columns={'time':'user_from'},inplace=True)
    user_to = letter_data.groupby(letter_data['user_to']).count()
    user_to = user_to.drop(['user_from'],axis=1)
    user_to.rename(columns={'time':'user_to'},inplace=True)
    user_data = user_data.join(user_from,how='outer')
    user_data = user_data.join(user_to,how='outer')
    print "私信记录处理完毕！"

    user_data = user_data.fillna(value=0)
    print user_data.iloc[:15]


    user_data.to_csv('user_data2015.txt',index=True,sep=' ')
    print "数据处理完毕！"

#验证处理的数据是否正确
def verify_data():
    user_data = pd.read_csv('user_data2015.txt',delimiter=' ')
    # print user_data.shape  # 157427
    # post_num browse_num comment_num voteup_num votedown_num favorite_num follow_num fan_num user_from user_to
    blog_num = user_data.loc[:,'post_num'].sum()
    browse_num = user_data.loc[:,'browse_num'].sum()
    comment_num = user_data.loc[:,'comment_num'].sum()
    voteup_num = user_data.loc[:,'voteup_num'].sum()
    votedown_num = user_data.loc[:,'votedown_num'].sum()
    favorite_num = user_data.loc[:,'favorite_num'].sum()
    follow_num = user_data.loc[:,'follow_num'].sum()
    fan_num = user_data.loc[:,'fan_num'].sum()
    user_from = user_data.loc[:,'user_from'].sum()
    user_to = user_data.loc[:,'user_to'].sum()

    print "博客实际1000000条，处理之后%s条，是否符合：%s"%(blog_num,blog_num==1000000)
    print "浏览实际3536444条，处理之后%s条，是否符合：%s"%(browse_num,browse_num==3536444)
    print "评论实际182273条，处理之后%s条，是否符合：%s"%(comment_num,comment_num==182273)
    print "点赞实际95668条，处理之后%s条，是否符合：%s"%(voteup_num,voteup_num==95668)
    print "踩实际9326条，处理之后%s条，是否符合：%s"%(votedown_num,votedown_num==9326)
    print "收藏实际104723条，处理之后%s条，是否符合：%s"%(favorite_num,favorite_num==104723)
    print "跟随实际667037条，处理之后%s条，是否符合：%s"%(follow_num,follow_num==667037)
    print "粉丝实际667037条，处理之后%s条，是否符合：%s"%(fan_num,fan_num==667037)
    print "发信实际46572条，处理之后%s条，是否符合：%s"%(user_from,user_from==46572)
    print "收信实际46572条，处理之后%s条，是否符合：%s"%(user_to,user_to==46572)

    print "验证完成！"

# 1015条数据
def gen_train_dataset():
    user_data = pd.read_csv('user_data2015.txt', delimiter=' ',index_col=0)  # 未制定index，导致join失败
    user_values = pd.read_csv('../data/SMPCUP2017_TrainingData_Task3.txt',delimiter='\001',header=None)
    user_values.rename(columns={0:'id',1:'value'},inplace=True)
    user_values = user_values.set_index(['id'])
    user_data = user_data.join(user_values,how='inner')
    user_data.to_csv('train_dataset.txt',index=True,sep=' ')
    print "训练集生成完毕！"

# 训练xgboost模型 在训练过程中，100多次已经达到了最佳
# 待训练的参数：学习率：eta,silent,
def train_xgb_model(usecols):
    data = np.loadtxt('train_dataset.txt',delimiter=' ',usecols=usecols,skiprows=1)
    # print data[1,:]
    # print data.shape
    use_col_num = data.shape[1]
    x,y = np.split(data,(use_col_num-1,),axis=1)  # 索引是后面的起点
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.2)
    test_num = y_test.shape[0]

    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_train, 'train'),(data_test, 'eval')]
    # 默认gbtree 改变参数损失值还是基本不变
    param = {'booster':'gbtree','max_depth': 7,'gamma':0.000001, 'eta': 0.2, 'silent': 1, 'objective': 'reg:logistic','eval_metric':'mae'}
    # reg:logistic0.30 和reg:linear效果很不好，误差0.648
    # mae 和 rmse 结果基本一样
    bst = xgb.train(param, data_train, num_boost_round=50000, evals=watch_list,early_stopping_rounds=1000)

    y_hat = bst.predict(data_test).reshape(test_num,)
    y_test = y_test.reshape(test_num,)
    print y_test
    print y_hat

    y_max = []
    for i in range(test_num):
        y_max.append(y_test[i] if y_test[i] > y_hat[i] else y_hat[i])

    result = (np.abs(y_test - y_hat)/(np.array(y_max))).sum()/test_num
    print 'result', result
    # print '正确率：\t', float(np.sum(result)) / len(y_hat)
    print 'END.....\n'
    return result

# 这个函数使用所有的train数据带入训练，没有test数据（无交叉验证）  1.0版本
def gen_valid_values(max_round=100):
    data = np.loadtxt('train_dataset.txt', delimiter=' ', usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), skiprows=1)
    x, y = np.split(data, (10,), axis=1)  # 索引是后面的起点
    valid_userID = []
    with open('../data/valid_task3.txt','r') as f:
        ID = f.readline()
        while ID:
            if ID[-1] == '\n':
                valid_userID.append(ID[:-1])
            else:
                valid_userID.append(ID)
            ID = f.readline()

    user_data = pd.read_csv('user_data2015.txt',delimiter=' ',index_col=0)
    # 不写loc的话keyerror
    user_data = user_data.loc[valid_userID]
    user_data = user_data.values

    data_train = xgb.DMatrix(x, label=y)
    valid_data = xgb.DMatrix(user_data,label=user_data[:,0])
    watch_list = [(data_train, 'train')]
    # 默认gbtree
    param = {'booster':'gbtree','max_depth': 10, 'eta': 0.01, 'silent': 1, 'objective': 'reg:logistic'}
    bst = xgb.train(param, data_train, num_boost_round=max_round, evals=watch_list)

    y_hat = bst.predict(valid_data).reshape(user_data.shape[0], )
    print y_hat
    with open('tesk3_result.txt','w') as f:
        for i in range(len(valid_userID)):
            f.write(valid_userID[i]+','+str(y_hat[i])+'\n')

    # print '正确率：\t', float(np.sum(result)) / len(y_hat)
    print 'END.....\n'

# 使用交叉验证预测
def gen_valid_values_2(max_round=50000):
    data = np.loadtxt('train_dataset.txt', delimiter=' ', usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), skiprows=1)
    x, y = np.split(data, (10,), axis=1)  # 索引是后面的起点
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.2)
    test_num = y_test.shape[0]

    valid_userID = []
    with open('../data/valid_task3.txt', 'r') as f:
        ID = f.readline()
        while ID:
            if ID[-1] == '\n':
                valid_userID.append(ID[:-1])
            else:
                valid_userID.append(ID)
            ID = f.readline()

    user_data = pd.read_csv('user_data2015.txt', delimiter=' ', index_col=0)
    # 不写loc的话keyerror
    user_data = user_data.loc[valid_userID]
    user_data = user_data.values

    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    valid_data = xgb.DMatrix(user_data, label=user_data[:, 0])
    watch_list = [(data_train, 'train'), (data_test, 'eval')]
    # 默认gbtree
    param = {'booster': 'gbtree', 'max_depth': 6, 'eta': 0.1, 'silent': 1, 'objective': 'reg:logistic',
             'eval_metric': 'mae'}
    # reg:logistic0.30 和reg:linear效果很不好，误差0.648
    # mae 和 rmse 结果基本一样
    bst = xgb.train(param, data_train, num_boost_round=50000, evals=watch_list, early_stopping_rounds=1000)

    y_hat = bst.predict(data_test).reshape(test_num, )
    y_test = y_test.reshape(test_num, )

    y_max = []
    for i in range(test_num):
        y_max.append(y_test[i] if y_test[i] > y_hat[i] else y_hat[i])

    result = (np.abs(y_test - y_hat) / (np.array(y_max))).sum() / test_num
    print '模型的相对误差：', result

    y_predict = bst.predict(valid_data).reshape(user_data.shape[0], )
    print y_predict
    with open('tesk3_result.txt', 'w') as f:
        for i in range(len(valid_userID)):
            f.write(valid_userID[i] + ',' + str(y_predict[i]) + '\n')


    print 'END.....\n'

# 通过分别计算单个特征预测的误差观察其各自的重要性
# 第1个特征训练的误差为：0.399081362548 1 post
# 第2个特征训练的误差为：0.529250131277 2 browse
# 第3个特征训练的误差为：0.719933047665 4 comment
# 第4个特征训练的误差为：0.721075074391 5 vote_up
# 第5个特征训练的误差为：0.780544591893 10 vote_down 差
# 第6个特征训练的误差为：0.778198078944 9 favorite
# 第7个特征训练的误差为：0.743326627439 7 follow 差
# 第8个特征训练的误差为：0.658186988878 3 fan
# 第9个特征训练的误差为：0.755958878322 8 letter_from 差
# 第10个特征训练的误差为：0.730409222336 6 letter_to

# 使用交叉验证 预测结果
def gen_test_values(max_round=50000):
    data = np.loadtxt('train_dataset.txt', delimiter=' ', usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), skiprows=1)
    x, y = np.split(data, (10,), axis=1)  # 索引是后面的起点
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.2)
    test_num = y_test.shape[0]

    valid_userID = []
    with open('../test/SMPCUP2017_TestSet_Task3.txt', 'rb') as f:
        for ID in f:
            ID = ID.strip()
            # print ID
            if len(ID) == 11:
                ID = ID[3:]
            if ID[-1] == '\n':
                valid_userID.append(ID[:-1])
            else:
                valid_userID.append(ID)

    # print valid_userID
    user_data = pd.read_csv('user_data2015.txt', delimiter=' ', index_col=0)
    # 不写loc的话keyerror
    user_data = user_data.loc[valid_userID]
    user_data = user_data.values

    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    valid_data = xgb.DMatrix(user_data, label=user_data[:, 0])
    watch_list = [(data_train, 'train'), (data_test, 'eval')]
    # 默认gbtree
    param = {'booster': 'gbtree', 'max_depth': 6, 'eta': 0.1, 'silent': 1, 'objective': 'reg:logistic',
             'eval_metric': 'mae'}
    # reg:logistic0.30 和reg:linear效果很不好，误差0.648
    # mae 和 rmse 结果基本一样
    bst = xgb.train(param, data_train, num_boost_round=50000, evals=watch_list, early_stopping_rounds=1000)

    y_hat = bst.predict(data_test).reshape(test_num, )
    y_test = y_test.reshape(test_num, )

    y_max = []
    for i in range(test_num):
        y_max.append(y_test[i] if y_test[i] > y_hat[i] else y_hat[i])

    result = (np.abs(y_test - y_hat) / (np.array(y_max))).sum() / test_num
    print '模型的相对误差：', result

    y_predict = bst.predict(valid_data).reshape(user_data.shape[0], )
    print y_predict
    with open('task3_result.txt', 'w') as f:
        for i in range(len(valid_userID)):
            f.write(valid_userID[i] + ',' + str(y_predict[i]) + '\n')


    print 'END.....\n'

gen_test_values()

def rank_ten_feature():
    pre_error = []
    for i in range(10):
        usecol = (i+1,11)
        pre_error.append(train_xgb_model(usecol))
    for i in range(10):
        print "第%s个特征训练的误差为：%s"%(str(i+1),pre_error[i])
    pre_error.sort()
    print pre_error

# process_useraction_data()
# print verify_data()
# gen_train_dataset()
# train_xgb_model((1,2,8,3,4,10,7,9,5,11))
# gen_valid_values_2(max_round=1000)
# rank_ten_feature()

# 使用不同的列进行预测
# (1) 0.399081362548
# (1,2) 0.283561067996
# (1,2,8) 0.280316678656
# (1,2,8,3) 0.28222014761
# (1,2,8,3,4) 0.282950003087
# (1,2,8,3,4,10) 0.279196011228   这一个预测效果最好
# (1,2,8,3,4,10,7) 0.289244687745
# (1,2,8,3,4,10,7,9) 0.288582146792
# (1,2,8,3,4,10,7,9,5) 0.292147639524

# 以下的内容是使用时间 分别使用one-hot编码将用户的行为数据编码为365维的向量
def process_three_cloumns_data_use_date(file_path,out_file):
    # '../data/task3/blog_date_vec.txt'
    date_start = datetime.datetime.strptime('2015/01/01','%Y/%m/%d')
    post_blog_by_date = {}
    with open(file_path,'r') as f:
        zero_vec = np.zeros(shape=(365,))
        # print len(vec),vec
        for lineno,line in enumerate(f,1):
            # if lineno == 1:
            #     print line
            # if lineno == 9326:
            #     print line
            if lineno % 500 == 0:
                # break
                print lineno
            user_id = line[:8]
            # print line[:8],line[18:-1]
            # print line[18:-1]
            days_delta = (datetime.datetime.strptime(line[18:-2],"%Y-%m-%d %H:%M:%S")-date_start).days  # .%f
            if user_id not in post_blog_by_date.keys():
                post_blog_by_date[user_id] = zero_vec.copy()
                post_blog_by_date[user_id][days_delta] += 1
            else:
                post_blog_by_date[user_id][days_delta] += 1
        print lineno

    with open(out_file,'w') as f:
        for key in post_blog_by_date.keys():
            f.write(key+','+','.join([str(int(item)) for item in post_blog_by_date[key]])+'\n')
    print "END..."

def process_letter_to_data_use_date(file_path,out_file):
    # '../data/task3/blog_date_vec.txt'
    date_start = datetime.datetime.strptime('2015/01/01','%Y/%m/%d')
    post_blog_by_date = {}
    with open(file_path,'r') as f:
        zero_vec = np.zeros(shape=(365,))
        # print len(vec),vec
        for lineno,line in enumerate(f,1):
            if lineno % 500 == 0:
                # break
                print lineno
            user_id = line[9:17]
            print user_id
            # print line[:8],line[18:-1]
            # print line[18:-1]
            days_delta = (datetime.datetime.strptime(line[18:-2],"%Y-%m-%d %H:%M:%S.%f")-date_start).days
            if user_id not in post_blog_by_date.keys():
                post_blog_by_date[user_id] = zero_vec.copy()
                post_blog_by_date[user_id][days_delta] += 1
            else:
                post_blog_by_date[user_id][days_delta] += 1
        print lineno

    with open(out_file,'w') as f:
        for key in post_blog_by_date.keys():
            f.write(key+','+','.join([str(int(item)) for item in post_blog_by_date[key]])+'\n')
    print "END..."

# process_three_cloumns_data_use_date(post_data_path,'../data/task3/blog_date_vec.txt')
# process_three_cloumns_data_use_date(comment_data_path,'../data/task3/comment_date_vec.txt')
# process_three_cloumns_data_use_date(browse_data_path,'../data/task3/browse_date_vec.txt')
# process_three_cloumns_data_use_date(voteup_data_path,'../data/task3/voteup_date_vec.txt')
# process_three_cloumns_data_use_date(votedown_data_path,'../data/task3/votedown_date_vec.txt')
# process_three_cloumns_data_use_date(letter_data_path,'../data/task3/letterfrom_date_vec.txt')
# process_letter_to_data_use_date(letter_data_path,'../data/task3/letterto_date_vec.txt')

def gen_trian_use_date(train_x_file,train_y_file,train_data_file):
    # '../data/SMPCUP2017_TrainingData_Task3.txt'
    # '../data/task3/blog_date_vec.txt'
    # '../data/task3/2_post_train_use_date.txt'
    train_data = {}
    with open(train_x_file,'r') as f:
        for line in f:
            id,value = line.split('\001')
            if id not in train_data.keys():
                train_data[id] = value
    # value后面自带换行

    train_data_use_date = {}
    with open(train_y_file,'r') as f:
        for line_no,line in enumerate(f,1):
            if line_no % 500 == 0:
                print line_no
            id = line[:8]
            vec = str(line[9:-1])
            if id in train_data.keys():
                vec += ","+str(train_data[id])
                train_data_use_date[id] = vec
    with open(train_data_file,'w') as f:
        for key in train_data_use_date.keys():
            f.write(key+","+train_data_use_date[key])

    print "END..."

# type表示是否使用导数以及几阶导
def train_xgbmodel_use_date(data_path,features=365,type=0):
    usecols = []
    for i in range(1,features+2):
        usecols.append(i)
    data = np.loadtxt(data_path, delimiter=',', usecols=tuple(usecols), skiprows=0)

    use_col_num = data.shape[1]
    x, y = np.split(data, (use_col_num - 1,), axis=1)  # 索引是后面的起点
    while type > 0:
        x = np.diff(x)
        type -= 1

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.2)
    test_num = y_test.shape[0]

    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_train, 'train'), (data_test, 'eval')]
    # 默认gbtree 改变参数损失值还是基本不变
    param = {'booster': 'gbtree', 'max_depth': 7, 'gamma': 0.000001, 'eta': 0.2, 'silent': 1,
             'objective': 'reg:logistic', 'eval_metric': 'mae'}
    # reg:logistic0.30 和reg:linear效果很不好，误差0.648
    # mae 和 rmse 结果基本一样
    bst = xgb.train(param, data_train, num_boost_round=50000, evals=watch_list, early_stopping_rounds=1000)

    y_hat = bst.predict(data_test).reshape(test_num, )
    y_test = y_test.reshape(test_num, )
    print y_test
    print y_hat

    y_max = []
    for i in range(test_num):
        y_max.append(y_test[i] if y_test[i] > y_hat[i] else y_hat[i])

    result = (np.abs(y_test - y_hat) / (np.array(y_max))).sum() / test_num
    print 'result', result
    # print '正确率：\t', float(np.sum(result)) / len(y_hat)
    print 'END.....\n'
    return result




#---------------------------------------------------------------
# 结果分析：
#  特征           365维             364维（一阶导数）    二阶导数          十阶导数
# blog         0.54677717733       0.573102533662   0.544426425501     0.497959267801
# browse       0.485429240938      0.484860274937   0.45726649773      0.452422810699
# comment      0.587040348256      0.629389720595   0.544209333401     0.53665981009
# vote_up      0.637330020769      0.62995187801    0.610429689971     0.599032886389
# vote_down    0.69850424465       0.69850424465    0.69850424465      0.69850424465
# favorite     0.668163645044      0.668163645044   0.668163645044     0.649685708747
# follow
# fan
# letter_from  0.655563472876      0.655563472876   0.655563472876     0.681748723907
# letter_to    0.744867129708      0.744867129708   0.738078446872     0.711026351143


# process_three_cloumns_data_use_date(post_data_path,'../data/task3/blog_date_vec.txt')
# process_three_cloumns_data_use_date(comment_data_path,'../data/task3/comment_date_vec.txt')
# process_three_cloumns_data_use_date(browse_data_path,'../data/task3/browse_date_vec.txt')
# process_three_cloumns_data_use_date(voteup_data_path,'../data/task3/voteup_date_vec.txt')
# process_three_cloumns_data_use_date(votedown_data_path,'../data/task3/votedown_date_vec.txt')
# process_three_cloumns_data_use_date(letter_data_path,'../data/task3/letterfrom_date_vec.txt')
# process_letter_to_data_use_date(letter_data_path,'../data/task3/letterto_date_vec.txt')
# gen_trian_use_date('../data/SMPCUP2017_TrainingData_Task3.txt','../data/task3/blog_date_vec.txt','../data/task3/2_postblog_train_use_date.txt')
# gen_trian_use_date('../data/SMPCUP2017_TrainingData_Task3.txt','../data/task3/browse_date_vec.txt','../data/task3/3_browse_train_use_date.txt')
# gen_trian_use_date('../data/SMPCUP2017_TrainingData_Task3.txt','../data/task3/comment_date_vec.txt','../data/task3/4_comment_train_use_date.txt')
# gen_trian_use_date('../data/SMPCUP2017_TrainingData_Task3.txt','../data/task3/voteup_date_vec.txt','../data/task3/5_voteup_train_use_date.txt')
# gen_trian_use_date('../data/SMPCUP2017_TrainingData_Task3.txt','../data/task3/votedown_date_vec.txt','../data/task3/6_votedown_train_use_date.txt')
# gen_trian_use_date('../data/SMPCUP2017_TrainingData_Task3.txt','../data/task3/favorite_date_vec.txt','../data/task3/7_favorite_train_use_date.txt')
# gen_trian_use_date('../data/SMPCUP2017_TrainingData_Task3.txt','../data/task3/letterfrom_date_vec.txt','../data/task3/10_letterfrom_train_use_date.txt')
# gen_trian_use_date('../data/SMPCUP2017_TrainingData_Task3.txt','../data/task3/letterto_date_vec.txt','../data/task3/11_letterto_train_use_date.txt')
# train_xgbmodel_use_date('../data/task3/2_postblog_train_use_date.txt',features=365,type=10)
# train_xgbmodel_use_date('../data/task3/3_browse_train_use_date.txt',features=365,type=10)
# train_xgbmodel_use_date('../data/task3/4_comment_train_use_date.txt',features=365,type=10)
# train_xgbmodel_use_date('../data/task3/5_voteup_train_use_date.txt',features=365,type=10)
# train_xgbmodel_use_date('../data/task3/6_votedown_train_use_date.txt',features=365,type=10)
# train_xgbmodel_use_date('../data/task3/7_favorite_train_use_date.txt',features=365,type=10)
# train_xgbmodel_use_date('../data/task3/10_letterfrom_train_use_date.txt',features=365,type=10)
# train_xgbmodel_use_date('../data/task3/11_letterto_train_use_date.txt',features=365,type=10)

# 使用不同的模式拟合模型，选取最好的权重
def train_weight(filename,features=365,type=0):
    usecols = []
    for i in range(1, features + 2):
        usecols.append(i)
    data = np.loadtxt(filename,delimiter=',',usecols=usecols,skiprows=0)

    use_col_num = data.shape[1]
    user_num = data.shape[0]
    x, y = np.split(data, (use_col_num - 1,), axis=1)  # 索引是后面的起点
    while type > 0:
        x = np.diff(x)
        type -= 1

    use_col_num = x.shape[1]

    result_score = []
    for iter_num in range(3):
        if iter_num == 0:
            weight = range(1,use_col_num+1)
            # print len(weight)
            # print x[0],len(x[0])
            x_ = [(np.array(item)*np.array(weight)).sum() for item in x.copy()]
            x_ = np.array(x_).reshape(user_num,-1)
            # print x_,x_.shape
        elif iter_num == 1:
            weight = [math.exp(item/100.0) for item in range(1, use_col_num+1)]
            x_ = [(np.array(item) * np.array(weight)).sum() for item in x.copy()]
            x_ = np.array(x_).reshape(user_num, -1)
        else:
            weight = [math.log(item) for item in range(1, use_col_num+1)]
            x_ = [(np.array(item) * np.array(weight)).sum() for item in x.copy()]
            x_ = np.array(x_).reshape(user_num, -1)


        x_train, x_test, y_train, y_test = train_test_split(x_, y, random_state=1, test_size=0.2)
        test_num = y_test.shape[0]

        data_train = xgb.DMatrix(x_train, label=y_train)
        data_test = xgb.DMatrix(x_test, label=y_test)
        watch_list = [(data_train, 'train'), (data_test, 'eval')]
        # 默认gbtree 改变参数损失值还是基本不变
        param = {'booster': 'gbtree', 'max_depth': 7, 'gamma': 0.000001, 'eta': 0.2, 'silent': 1,
                 'objective': 'reg:logistic', 'eval_metric': 'mae'}
        # reg:logistic0.30 和reg:linear效果很不好，误差0.648
        # mae 和 rmse 结果基本一样
        bst = xgb.train(param, data_train, num_boost_round=50000, evals=watch_list, early_stopping_rounds=1000)

        y_hat = bst.predict(data_test).reshape(test_num, )
        y_test = y_test.reshape(test_num, )
        # print y_test
        # print y_hat

        y_max = []
        for i in range(test_num):
            y_max.append(y_test[i] if y_test[i] > y_hat[i] else y_hat[i])

        result = (np.abs(y_test - y_hat) / (np.array(y_max))).sum() / test_num
        result_score.append(result)
    for i in range(len(result_score)):
        print '第%s次的result：%s'%(i,result_score[i])
    print result_score[0],result_score[1],result_score[2]
    print 'END.....\n'
    return result

# 使用不同的函数对趋势归一化：求导多次之后效果反而不好 一阶导数的效果最好
# 感觉使用原数据和一阶导数的对数效果比较好
#  特征        自然整数       幂次方           对数          导数
# postblog 0.505927078268 0.552599643987 0.441998866681 0.435747385268 0.541687628273 0.505024869027
# comment  0.537955797055 0.575336268455 0.553795678251 0.548726887958 0.5945620535 0.508136748586
# voteup   0.622695130225 0.645510423962 0.622686335589 0.57844697118 0.644645884826 0.673675120617
# votedown 0.69943045181 0.701882648838 0.69850424465   0.69850424465 0.704313768612 0.69850424465
# favorite 0.599944338343 0.635470445443 0.645969380801 0.672933495854 0.641556729987 0.625615729013
# letterfrom 0.648720608707 0.678667855581 0.661141655646 0.718945133686 0.677772153506 0.665452523475
# letterto 0.691037414198 0.762865177232 0.687282901444 0.747105148428 0.76520394332 0.66614388483
# train_weight('../data/task3/2_postblog_train_use_date.txt',type=2)
# train_weight('../data/task3/4_comment_train_use_date.txt',type=1)
# train_weight('../data/task3/5_voteup_train_use_date.txt',type=0)
# train_weight('../data/task3/6_votedown_train_use_date.txt',type=1)
# train_weight('../data/task3/7_favorite_train_use_date.txt',type=1)
# train_weight('../data/task3/10_letterfrom_train_use_date.txt',type=1)
# train_weight('../data/task3/11_letterto_train_use_date.txt',type=1)
