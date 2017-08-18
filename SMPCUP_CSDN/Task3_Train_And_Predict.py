#-*- coding:UTF-8 -*-
# author:jaylin
# time:17-7-6 下午3:19
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import math,datetime


post_file = '../data/task3/2_postblog_train_use_date.txt'
browse_file = '../data/task3/3_browse_train_use_date.txt'
comment_file = '../data/task3/4_comment_train_use_date.txt'
voteup_file = '../data/task3/5_voteup_train_use_date.txt'
votedown_file = '../data/task3/6_votedown_train_use_date.txt'
favorite_file = '../data/task3/7_favorite_train_use_date.txt'
letterfrom_file = '../data/task3/10_letterfrom_train_use_date.txt'
letterto_file = '../data/task3/11_letterto_train_use_date.txt'

postblog_vec = '../data/task3/blog_date_vec.txt'
browse_vec = '../data/task3/browse_date_vec.txt'
comment_vec = '../data/task3/comment_date_vec.txt'
favorite_vec = '../data/task3/favorite_date_vec.txt'
letterfrom_vec = '../data/task3/letterfrom_date_vec.txt'
letterto_vec = '../data/task3/letterto_date_vec.txt'
voteup_vec = '../data/task3/voteup_date_vec.txt'
votedown_vec = '../data/task3/votedown_date_vec.txt'

origin_post_data = '../data/2_Post.txt'
origin_browse_data = '../data/3_Browse.txt'
origin_comment_data = '../data/4_Comment.txt'
origin_voteup_data = '../data/5_Vote-up.txt'
origin_votedown_data = '../data/6_Vote-down.txt'
origin_favorite_data = '../data/7_Favorite.txt'

passive_browse_vec = '../data/task3/passive/3_passive_browse_vec.txt'
passive_comment_vec = '../data/task3/passive/4_passive_comment_vec.txt'
passive_voteup_vec = '../data/task3/passive/5_passive_voteup_vec.txt'
passive_votedown_vec = '../data/task3/passive/6_passive_votedown_vec.txt'
passive_favorite_vec = '../data/task3/passive/7_passive_favorite_vec.txt'

passive_browse_train_data = '../data/task3/passive/3_passive_browse_train_data.txt'
passive_comment_train_data = '../data/task3/passive/4_passive_comment_train_data.txt'
passive_voteup_train_data = '../data/task3/passive/5_passive_voteup_train_data.txt'
passive_votedown_train_data = '../data/task3/passive/6_passive_votedown_train_data.txt'
passive_favorite_train_data = '../data/task3/passive/7_passive_favorite_train_data.txt'

train_user_values = '../data/SMPCUP2017_TrainingData_Task3.txt'

# 生成用户的被动数据 被浏览,被评论,被点赞,被踩,被收藏
def gen_use_passive_data():
    # 1000000篇博客和对应的用户
    user_post_blog_dic = {}
    with open(origin_post_data,'rb') as f:
        for line in f:
            user_id = line[:8]
            blog_id = line[9:17]
            user_post_blog_dic[blog_id] = user_id
    return user_post_blog_dic
    # print len(user_post_blog_dic.keys())

# 以下的内容是使用时间的被动数据 分别使用one-hot编码将用户的行为数据编码为365维的向量
def process_three_cloumns_data_passive(file_path,out_file):
    # '../data/task3/blog_date_vec.txt'
    user_blog_dic = gen_use_passive_data()
    print "博客用户对应词典长度:",len(user_blog_dic.keys())
    date_start = datetime.datetime.strptime('2015/01/01','%Y/%m/%d')
    passive_data_vec = {}
    # print passive_data_vec
    with open(file_path,'rb') as f:
        zero_vec = np.zeros(shape=(365,))

        for lineno,line in enumerate(f,1):
            if lineno % 500 == 0:
                # break
                print lineno
            blog_id = line[9:17]
            user_id = user_blog_dic[blog_id]
            days_delta = (datetime.datetime.strptime(line[18:-2],"%Y-%m-%d %H:%M:%S")-date_start).days
            if user_id not in passive_data_vec.keys():
                passive_data_vec[user_id] = zero_vec.copy()
                passive_data_vec[user_id][days_delta] += 1
            else:
                passive_data_vec[user_id][days_delta] += 1

    with open(out_file,'w') as f:
        num = 0
        for key in passive_data_vec.keys():
            num += passive_data_vec[key].sum()
            f.write(key+','+','.join([str(int(item)) for item in passive_data_vec[key]])+'\n')
    print "处理结果是否正确:%s"%(num==lineno)
    print "END..."



# gen_use_passive_data()
# process_three_cloumns_data_passive(origin_comment_data,passive_comment_vec)
# process_three_cloumns_data_passive(origin_browse_data,passive_browse_vec)
# process_three_cloumns_data_passive(origin_voteup_data,passive_voteup_vec)
# process_three_cloumns_data_passive(origin_votedown_data,passive_votedown_vec)
# process_three_cloumns_data_passive(origin_favorite_data,passive_favorite_vec)

def gen_train_data_passive(train_x_file,train_y_file,train_data_file):
    # '../data/SMPCUP2017_TrainingData_Task3.txt'
    # '../data/task3/blog_date_vec.txt'
    # '../data/task3/2_post_train_use_date.txt'
    train_data = {}
    with open(train_y_file,'r') as f:
        for line in f:
            id,value = line.split('\001')
            if id not in train_data.keys():
                train_data[id] = value
    # value后面自带换行

    train_data_use_date = {}
    with open(train_x_file,'r') as f:
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

# gen_train_data_passive(passive_browse_vec,train_user_values,passive_browse_train_data)
# gen_train_data_passive(passive_comment_vec,train_user_values,passive_comment_train_data)
# gen_train_data_passive(passive_voteup_vec,train_user_values,passive_voteup_train_data)
# gen_train_data_passive(passive_votedown_vec,train_user_values,passive_votedown_train_data)
# gen_train_data_passive(passive_favorite_vec,train_user_values,passive_favorite_train_data)

def gen_passive_num_feature(filename):
    dataFrame = pd.read_csv(filename,delimiter=',',header=None,index_col=0)
    dataFrame = dataFrame.sum(axis=1)
    # dataFrame.rename(columns={0:"num"},inplace=True)
    a = dataFrame.index
    dataFrame = pd.DataFrame(dataFrame.values,columns=['num'])
    dataFrame['id'] = a
    dataFrame.set_index('id',inplace=True)
    return dataFrame

# gen_passive_num_feature(passive_favorite_vec)


# 使用单个文件中的特征训练模型,返回结果
def train_weight(train_filename,alluser_filename,features=365,type=0,date_num=1):
    ids = []
    with open(alluser_filename,'rb') as f:
        for line in f:
            if len(line)>=8:
                ids.append(line[:8])
    usecols = []
    for i in range(1, features + 2):
        usecols.append(i)
    usecols_ = []
    for i in range(1, features + 1):
        usecols_.append(i)
    data = np.loadtxt(train_filename,delimiter=',',usecols=usecols,skiprows=0)
    all_user_data = np.loadtxt(alluser_filename,delimiter=',',usecols=usecols_,skiprows=0)

    use_col_num = data.shape[1]
    user_num = data.shape[0]
    user_num_ = all_user_data.shape[0]
    x, y = np.split(data, (use_col_num - 1,), axis=1)  # 索引是后面的起点
    if date_num != 1:
        ex_num = features % date_num
        ex_days = x[:,-1*date_num:-1*ex_num]
        x = np.hstack((x,ex_days))
        num = x.shape[1]/date_num
        x = x.reshape(-1,date_num)
        x = x.sum(axis=1)
        x = x.reshape(-1,num)

        ex_days = all_user_data[:, -1 * date_num:-1 * ex_num]
        all_user_data = np.hstack((all_user_data, ex_days))
        num = all_user_data.shape[1] / date_num
        all_user_data = all_user_data.reshape(-1, date_num)
        all_user_data = all_user_data.sum(axis=1)
        all_user_data = all_user_data.reshape(-1, num)

    while type > 0:
        x = np.diff(x)
        all_user_data = np.diff(all_user_data)
        type -= 1

    use_col_num = x.shape[1]

    weight = [math.log(item) for item in range(1, use_col_num+1)]
    x_ = [(np.array(item)*np.array(weight)).sum() for item in x]
    x_ = np.array(x_).reshape(user_num,-1)
    all_user_data = [(np.array(item) * np.array(weight)).sum() for item in all_user_data]
    all_user_data = np.array(all_user_data).reshape(user_num_, -1)

    x_train, x_test, y_train, y_test = train_test_split(x_, y, random_state=1, test_size=0.2)
    test_num = y_test.shape[0]


    dataset = xgb.DMatrix(all_user_data)
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

    y_predict = bst.predict(dataset).reshape(user_num_,)
    dataFrame = pd.DataFrame()
    dataFrame['id'] = ids
    dataFrame['x_'] = all_user_data.reshape(1,-1)[0]
    dataFrame['y_predict'] = y_predict

    dataFrame.set_index('id',inplace=True)

    # print dataFrame
    #
    # print result
    print 'END.....\n'
    return result,dataFrame


def gen_all_passive_data():
    dataFrame_use_date = pd.DataFrame()
    data_ = gen_passive_num_feature(passive_browse_vec)
    data_.rename(columns={'num':'b_p_num'},inplace=True)
    # print data_
    dataFrame_use_date = dataFrame_use_date.join(data_,how='outer')
    # print dataFrame_use_date.describe()
    data_ = gen_passive_num_feature(passive_comment_vec)
    data_.rename(columns={'num': 'c_p_num'},inplace=True)
    dataFrame_use_date = dataFrame_use_date.join(data_, how='outer')
    data_ = gen_passive_num_feature(passive_voteup_vec)
    data_.rename(columns={'num': 'vu_p_num'},inplace=True)
    dataFrame_use_date = dataFrame_use_date.join(data_, how='outer')
    data_ = gen_passive_num_feature(passive_votedown_vec)
    data_.rename(columns={'num': 'vd_p_num'},inplace=True)
    dataFrame_use_date = dataFrame_use_date.join(data_, how='outer')
    data_ = gen_passive_num_feature(passive_favorite_vec)
    data_.rename(columns={'num': 'f_p_num'},inplace=True)
    dataFrame_use_date = dataFrame_use_date.join(data_, how='outer')

    print dataFrame_use_date.describe()

    dates = [1, 7, 30]
    for type in range(0, 2):
        for date_num in dates:
            result_score, dataFrame = train_weight(passive_browse_train_data, passive_browse_vec, type=type, date_num=date_num)
            dataFrame.rename(columns={"x_": "b_p_x_" + str(type) + "_" + str(date_num),
                                      "y_predict": "b_p_y_" + str(type) + "_" + str(date_num)}, inplace=True)

            dataFrame_use_date = dataFrame_use_date.join(dataFrame, how='outer')



            result_score, dataFrame = train_weight(passive_comment_train_data, passive_comment_vec, type=type, date_num=date_num)
            dataFrame.rename(columns={"x_": "c_p_x_" + str(type) + "_" + str(date_num),
                                      "y_predict": "c_p_y_" + str(type) + "_" + str(date_num)}, inplace=True)
            dataFrame_use_date = dataFrame_use_date.join(dataFrame, how='outer')


            result_score, dataFrame = train_weight(passive_voteup_train_data, passive_voteup_vec, type=type, date_num=date_num)
            dataFrame.rename(columns={"x_": "vu_p_x_" + str(type) + "_" + str(date_num),
                                      "y_predict": "vu_p_y_" + str(type) + "_" + str(date_num)}, inplace=True)
            dataFrame_use_date = dataFrame_use_date.join(dataFrame, how='outer')

            result_score, dataFrame = train_weight(passive_votedown_train_data, passive_votedown_vec, type=type, date_num=date_num)
            dataFrame.rename(columns={"x_": "vd_p_x_" + str(type) + "_" + str(date_num),
                                      "y_predict": "vd_p_y_" + str(type) + "_" + str(date_num)}, inplace=True)
            dataFrame_use_date = dataFrame_use_date.join(dataFrame, how='outer')

            result_score, dataFrame = train_weight(passive_favorite_train_data, passive_favorite_vec, type=type, date_num=date_num)
            dataFrame.rename(columns={"x_": "f_p_x_" + str(type) + "_" + str(date_num),
                                      "y_predict": "f_p_y_" + str(type) + "_" + str(date_num)}, inplace=True)
            dataFrame_use_date = dataFrame_use_date.join(dataFrame, how='outer')


    dataFrame_use_date.fillna(value=0)
    print dataFrame.describe()
    dataFrame_use_date.to_csv('../data/task3/passive/all_data_passive.txt', index=True, sep=',')

# gen_all_passive_data()

# 该文件用来构造特征训练模型拟合task3
# 主要用来选取特征 时间序列的选取分别为:1天 7天 30天
#  特征          1天      7天            30天           导数:1天        7天            30天
#postblog 0.441998866681 0.458440173769 0.492602458698 0.505024869027 0.482121612996 0.511657946151 0.58590846169
def gen_dataset_use_date():
    dataFrame_use_date = pd.DataFrame()
    dates = [1,7,30]
    for type in range(0,2):
        for date_num in dates:
            result_score, dataFrame = train_weight(post_file,postblog_vec,type=type,date_num=date_num)
            dataFrame.rename(columns={"x_":"p_x_"+str(type)+"_"+str(date_num),"y_predict":"p_y_"+str(type)+"_"+str(date_num)},inplace=True)

            dataFrame_use_date = dataFrame_use_date.join(dataFrame, how='outer')
            # print dataFrame_use_date
            # break


            result_score, dataFrame = train_weight(comment_file,comment_vec,type=type,date_num=date_num)
            dataFrame.rename(columns={"x_": "c_x_" + str(type) + "_" + str(date_num),
                                      "y_predict": "c_y_" + str(type) + "_" + str(date_num)}, inplace=True)
            dataFrame_use_date = dataFrame_use_date.join(dataFrame, how='outer')
            # print dataFrame
            # print dataFrame_use_date
            # break

            result_score, dataFrame = train_weight(browse_file,browse_vec,type=type,date_num=date_num)
            dataFrame.rename(columns={"x_": "b_x_" + str(type) + "_" + str(date_num),
                                      "y_predict": "b_y_" + str(type) + "_" + str(date_num)}, inplace=True)
            dataFrame_use_date = dataFrame_use_date.join(dataFrame, how='outer')

            result_score, dataFrame = train_weight(voteup_file,voteup_vec,type=type,date_num=date_num)
            dataFrame.rename(columns={"x_": "vu_x_" + str(type) + "_" + str(date_num),
                                      "y_predict": "vu_y_" + str(type) + "_" + str(date_num)}, inplace=True)
            dataFrame_use_date = dataFrame_use_date.join(dataFrame, how='outer')

            result_score, dataFrame = train_weight(votedown_file,votedown_vec,type=type,date_num=date_num)
            dataFrame.rename(columns={"x_": "vd_x_" + str(type) + "_" + str(date_num),
                                      "y_predict": "vd_y_" + str(type) + "_" + str(date_num)}, inplace=True)
            dataFrame_use_date = dataFrame_use_date.join(dataFrame, how='outer')

            result_score, dataFrame = train_weight(favorite_file,favorite_vec,type=type,date_num=date_num)
            dataFrame.rename(columns={"x_": "f_x_" + str(type) + "_" + str(date_num),
                                      "y_predict": "f_y_" + str(type) + "_" + str(date_num)}, inplace=True)
            dataFrame_use_date = dataFrame_use_date.join(dataFrame, how='outer')

            result_score, dataFrame = train_weight(letterfrom_file,letterfrom_vec,type=type,date_num=date_num)
            dataFrame.rename(columns={"x_": "lf_x_" + str(type) + "_" + str(date_num),
                                      "y_predict": "lf_y_" + str(type) + "_" + str(date_num)}, inplace=True)
            dataFrame_use_date = dataFrame_use_date.join(dataFrame, how='outer')

            result_score, dataFrame = train_weight(letterto_file,letterto_vec,type=type,date_num=date_num)
            dataFrame.rename(columns={"x_": "lt_x_" + str(type) + "_" + str(date_num),
                                      "y_predict": "lt_y_" + str(type) + "_" + str(date_num)}, inplace=True)
            dataFrame_use_date = dataFrame_use_date.join(dataFrame, how='outer')
        # break

    dataFrame_use_date.fillna(value=0)
    print dataFrame_use_date.describe()
    dataFrame_use_date.to_csv('../data/task3/data_use_date.txt',index=True,sep=',')

# gen_dataset_use_date()
passive_data = '../data/task3/passive/all_data_passive.txt'
data_use_date = '../data/task3/data_use_date.txt'
num_data = '../data/task3/user_data2015.txt'
all_data = '../data/task3/all_data_task3.txt'
all_train_data = '../data/task3/all_train_data_task3.txt'

def gen_dataset_use_num_and_date_passive():
    data_use_date_ = pd.read_csv(data_use_date,delimiter=',',index_col=0)
    num_data_ = pd.read_csv(num_data,delimiter=' ',index_col=0)
    data_passive_ = pd.read_csv(passive_data,delimiter=',',index_col=0)
    all_data_ = data_use_date_.join(num_data_,how='outer')
    all_data_ = all_data_.join(data_passive_,how='outer')
    all_data_.to_csv(all_data,index=True,sep=',')
    all_data_ = all_data_.fillna(value=0)
    print all_data_.describe()
    print "END..."

# gen_dataset_use_num_and_date_passive()

# 1015条数据
def gen_train_dataset():
    user_data = pd.read_csv(all_data, delimiter=',',index_col=0)  # 未制定index，导致join失败
    user_values = pd.read_csv('../data/SMPCUP2017_TrainingData_Task3.txt',delimiter='\001',header=None)
    user_values.rename(columns={0:'id',1:'value'},inplace=True)
    user_values = user_values.set_index(['id'])
    user_data = user_data.join(user_values,how='inner')
    user_data = user_data.fillna(value=0)
    print user_data.describe()
    user_data.to_csv(all_train_data,index=True,sep=' ')
    print "训练集生成完毕！"

# gen_train_dataset()

# 使用交叉验证预测
def gen_valid_values_2(max_round=50000):
    use_clos =  tuple(range(1,173))

    data = np.loadtxt(all_train_data, delimiter=' ', usecols=use_clos, skiprows=1)
    x, y = np.split(data, (171,), axis=1)  # 索引是后面的起点
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

    user_data = pd.read_csv(all_data, delimiter=',', index_col=0)
    user_data = user_data.fillna(value=0)
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

# 使用交叉验证预测 使用不同的特征
def gen_valid_values_2_DF(max_round=50000,feature_list=None,feature_list_index=None):
    # use_clos =  tuple(range(1,173))
    #
    # data = np.loadtxt(all_train_data, delimiter=' ', usecols=use_clos, skiprows=1)
    data = pd.read_csv(all_train_data,delimiter=' ',index_col=0)
    if feature_list != None:
        feature_list.append('value')
        data = data[feature_list]
    if feature_list_index != None:
        feature_list_index.append(171)
        data = data.iloc[:,feature_list_index]
    print data.describe()
    train_index = data.index
    data = data.values
    x, y = np.split(data, (data.shape[1]-1,), axis=1)  # 索引是后面的起点
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

    user_data = pd.read_csv(all_data, delimiter=',', index_col=0)
    user_data = user_data.fillna(value=0)
    if feature_list != None:
        feature_list.remove('value')
        user_data = user_data[feature_list]
    if feature_list_index != None:
        feature_list_index.remove(171)
        user_data = user_data.iloc[:, feature_list_index]

    # 不写loc的话keyerror
    user_data = user_data.loc[valid_userID]
    user_data = user_data.values

    user_data_train = xgb.DMatrix(data[:,:-1])
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

    y_train_hat = bst.predict(user_data_train).reshape(data.shape[0], )
    print y_predict
    print len(train_index)
    print len(y_train_hat)
    with open('../data/task3/predict_train.txt','w') as f:
        for i in range(len(train_index)):
            f.write(train_index[i]+","+str(y_train_hat[i])+'\n')


    with open('tesk3_result.txt', 'w') as f:
        for i in range(len(valid_userID)):
            pre = y_predict[i]
            if y_predict[i]<0.0006:
                pre = 0.0

            f.write(valid_userID[i] + ',' + str(pre) + '\n')


    print 'END.....\n'

# 相关系数计算与排序
def cal_correlation():
    corrs = []
    data = pd.read_csv(all_train_data, delimiter=' ', index_col=0)
    for i in range(171):
        corr = data.iloc[:,i].corr(data.iloc[:,171])
        corrs.append(corr)
        print '第%s个特征的相关系数为:%s'%((i+1),corr)

    a = zip(corrs, range(len(corrs)))
    # print a
    a = sorted(a,key=lambda x: x[0],reverse=True)
    corrs = [x[0] for x in a]
    corrs_index_sort = [x[1] for x in a]
    # print corrs
    # print corrs_index_sort
    return corrs,corrs_index_sort

# [0.95408933803931806, 0.94035932375016018, 0.93098161548883707, 0.92274120277442195, 0.90165134075295339, 0.88828155196951764, 0.8865384872689358, 0.88574713912000691, 0.86961218937339835, 0.8638878117417359, 0.85364507676060419, 0.8438303038943481, 0.840751115504071, 0.83614197881809671, 0.83294571889244473, 0.82196978624864181, 0.81845960534919115, 0.81751178528885771, 0.81553383232750809, 0.81444710603351456, 0.80986401886741377, 0.79642684108728146, 0.77626493289418375, 0.7713454227974097, 0.76676751264589205, 0.76524097115083212, 0.75730199289853339, 0.74957506756297143, 0.74613290305869973, 0.73600797109334448, 0.73169449576326884, 0.73094734690973284, 0.72848267624420526, 0.7270711834771526, 0.72599479216281615, 0.71349262983477124, 0.70589951652715388, 0.69882887890769207, 0.68732654228489354, 0.67663332264697207, 0.63272534076314568, 0.62705310510385059, 0.62104806182595707, 0.61941290768182899, 0.61439777463235268, 0.61353594100504472, 0.60907854869850542, 0.59948089748538069, 0.58271989200343999, 0.57339925438181916, 0.57305364932358538, 0.56720389462585963, 0.5660453978201333, 0.56545800553134906, 0.56506989665219642, 0.56323673896385962, 0.56242271967508339, 0.56240569330173806, 0.56122135482192237, 0.56064326471212889, 0.55962512604441905, 0.55889053415813461, 0.5578771549520789, 0.54623034709497398, 0.54515348854570822, 0.54408993964389585, 0.53183866452547546, 0.5229467356727322, 0.49870844467928988, 0.46942087702522362, 0.4576194454740547, 0.45662561400607443, 0.45300628780486579, 0.45105460254829033, 0.44689554603079551, 0.44115398653836491, 0.41154880001570382, 0.40496978610224477, 0.39325221922161407, 0.39284803158507364, 0.38326667155452265, 0.38038346011946705, 0.35762203407516419, 0.34682192691033065, 0.34665087083764479, 0.34620896313956429, 0.34006684054213782, 0.33790053642684686, 0.33464911887839183, 0.33435965398545137, 0.32348014960232818, 0.29853235561200281, 0.28127829129063264, 0.2713037000624583, 0.21603800610536666, 0.20947241698782251, 0.20881957814922106, 0.20820578839954881, 0.2012945719479404, 0.20059767356369826, 0.19622904013620815, 0.19210934410727307, 0.19185122460313003, 0.18944440327307899, 0.18284453771503884, 0.17951959299448969, 0.1778273922553085, 0.17301701348624215, 0.17054277638809162, 0.15656352429458634, 0.15394519271949031, 0.14435042835249759, 0.13243849355316906, 0.13158144818144066, 0.12948467861169827, 0.12819540907132557, 0.12528470408859591, 0.12223064503618114, 0.11718477285834945, 0.11680041109993269, 0.097849747911527937, 0.095986980615321113, 0.094123952979439651, 0.093277496044191582, 0.092516689523457082, 0.090994026634009198, 0.090608028812994693, 0.090546693438751991, 0.089799956015637436, 0.088293062156895308, 0.088006583819330728, 0.087815255402977646, 0.087220877197025176, 0.086648297514724804, 0.085641447106788948, 0.085641447106788948, 0.085641447106788948, 0.085641447106788948, 0.085641447106788948, 0.085641447106788948, 0.084012864463609796, 0.083724882448298246, 0.083685930693360028, 0.082673071111457677, 0.082177335854618866, 0.081396192288234706, 0.07920884913725601, 0.078264936154609097, 0.074873918306017515, 0.074752523392866824, 0.063650204904313015, 0.056751208123973988, 0.053418302788745708, 0.044653435955511682, 0.044458994715454753, 0.040010508873757449, 0.037472631084346987, 0.0082022255608358886, 0.0072699922390287962, -0.0048018386994239104, -0.0070060323186668452, -0.010895880330213378, -0.020736033797986522, -0.028747243195871758, -0.030453321029221572, -0.043749821627929841, -0.054094425720053775, -0.063532212490196388, -0.068711177966265621, -0.087938076406901849, -0.096636212210379754]
# [37, 5, 1, 17, 49, 21, 85, 53, 33, 122, 112, 132, 65, 146, 152, 116, 136, 69, 142, 81, 126, 156, 96, 0, 166, 16, 150, 120, 32, 124, 134, 154, 162, 170, 144, 114, 160, 164, 130, 140, 97, 19, 3, 4, 20, 35, 51, 36, 7, 63, 23, 39, 83, 133, 123, 34, 113, 98, 2, 107, 79, 18, 67, 143, 71, 153, 15, 31, 87, 55, 47, 148, 158, 95, 118, 128, 13, 29, 66, 138, 168, 61, 163, 22, 77, 38, 84, 6, 82, 68, 99, 93, 45, 141, 50, 121, 135, 131, 111, 161, 125, 106, 102, 80, 115, 11, 59, 75, 108, 151, 52, 27, 129, 139, 43, 119, 110, 169, 103, 159, 165, 91, 137, 42, 127, 12, 28, 104, 117, 10, 109, 26, 44, 101, 9, 25, 41, 57, 73, 89, 30, 14, 105, 24, 8, 46, 70, 100, 90, 40, 86, 94, 149, 58, 155, 88, 64, 74, 60, 147, 56, 78, 72, 92, 157, 167, 48, 76, 145, 62, 54]

# cal_correlation()

def cal_score(a,b):
    num = len(a)
    max = []
    for i in range(num):
        max.append(a[i] if a[i]>b[i] else b[i])
    result = (np.abs(a - b) / (np.array(max))).sum() / num
    return result


# 直接使用一个模型对数据的拟合一直效果不好  尝试先使用二分类=>大数值 小数值 然后训练模型
#  值      大于的误差    小于的误差       大于的数量 小于的数量
# 0.02 0.0292628882203 0.214877051895 199        816
# 0.01 0.0441114364812 0.240073909289 319        696
# 0.009 0.0493432844902 0.240700299592 330       685
# 0.008 0.0536768278548 0.246788121803 359       656
# 0.007 0.0632453682461 0.260828037727 423       592
# 0.006 0.0771821388999 0.278401548135 504       511
# 0.005..
# 0.004..
# 0.003 0.0817174241303 0.280141269807 520       495
def view_train_value(train_filename,threshold=0.01):
    train_data = pd.read_csv(train_filename,index_col=0,header=None,delimiter='')
    predict_train_data = pd.read_csv('../data/task3/predict_train.txt',index_col=0,header=None,delimiter=',')

    values = train_data.values

    print "大于%s的数有:%s"%(str(threshold),len(values[values>threshold]))
    a = train_data[values > threshold].values
    b = predict_train_data[values > threshold].values
    print cal_score(a,b)

    print "小于等于%s的数有:%s" %(str(threshold),len(values[values <= threshold]))
    a = train_data[values <= threshold].values
    b = predict_train_data[values <= threshold].values
    print cal_score(a, b)
    # print train_data

view_train_value(train_user_values,threshold=0.003)
#
# gen_dataset_use_date()
# gen_dataset_use_num_and_date()
# gen_train_dataset()
feature_list = ['post_num', 'browse_num', 'comment_num', 'voteup_num', 'votedown_num',
                'favorite_num', 'follow_num', 'fan_num', 'user_from', 'user_to',
                'b_p_num', 'c_p_num', 'vu_p_num', 'vd_p_num', 'f_p_num']

corrs = [0.95408933803931806, 0.94035932375016018, 0.93098161548883707, 0.92274120277442195, 0.90165134075295339, 0.88828155196951764, 0.8865384872689358, 0.88574713912000691, 0.86961218937339835, 0.8638878117417359, 0.85364507676060419, 0.8438303038943481, 0.840751115504071, 0.83614197881809671, 0.83294571889244473, 0.82196978624864181, 0.81845960534919115, 0.81751178528885771, 0.81553383232750809, 0.81444710603351456, 0.80986401886741377, 0.79642684108728146, 0.77626493289418375, 0.7713454227974097, 0.76676751264589205, 0.76524097115083212, 0.75730199289853339, 0.74957506756297143, 0.74613290305869973, 0.73600797109334448, 0.73169449576326884, 0.73094734690973284, 0.72848267624420526, 0.7270711834771526, 0.72599479216281615, 0.71349262983477124, 0.70589951652715388, 0.69882887890769207, 0.68732654228489354, 0.67663332264697207, 0.63272534076314568, 0.62705310510385059, 0.62104806182595707, 0.61941290768182899, 0.61439777463235268, 0.61353594100504472, 0.60907854869850542, 0.59948089748538069, 0.58271989200343999, 0.57339925438181916, 0.57305364932358538, 0.56720389462585963, 0.5660453978201333, 0.56545800553134906, 0.56506989665219642, 0.56323673896385962, 0.56242271967508339, 0.56240569330173806, 0.56122135482192237, 0.56064326471212889, 0.55962512604441905, 0.55889053415813461, 0.5578771549520789, 0.54623034709497398, 0.54515348854570822, 0.54408993964389585, 0.53183866452547546, 0.5229467356727322, 0.49870844467928988, 0.46942087702522362, 0.4576194454740547, 0.45662561400607443, 0.45300628780486579, 0.45105460254829033, 0.44689554603079551, 0.44115398653836491, 0.41154880001570382, 0.40496978610224477, 0.39325221922161407, 0.39284803158507364, 0.38326667155452265, 0.38038346011946705, 0.35762203407516419, 0.34682192691033065, 0.34665087083764479, 0.34620896313956429, 0.34006684054213782, 0.33790053642684686, 0.33464911887839183, 0.33435965398545137, 0.32348014960232818, 0.29853235561200281, 0.28127829129063264, 0.2713037000624583, 0.21603800610536666, 0.20947241698782251, 0.20881957814922106, 0.20820578839954881, 0.2012945719479404, 0.20059767356369826, 0.19622904013620815, 0.19210934410727307, 0.19185122460313003, 0.18944440327307899, 0.18284453771503884, 0.17951959299448969, 0.1778273922553085, 0.17301701348624215, 0.17054277638809162, 0.15656352429458634, 0.15394519271949031, 0.14435042835249759, 0.13243849355316906, 0.13158144818144066, 0.12948467861169827, 0.12819540907132557, 0.12528470408859591, 0.12223064503618114, 0.11718477285834945, 0.11680041109993269, 0.097849747911527937, 0.095986980615321113, 0.094123952979439651, 0.093277496044191582, 0.092516689523457082, 0.090994026634009198, 0.090608028812994693, 0.090546693438751991, 0.089799956015637436, 0.088293062156895308, 0.088006583819330728, 0.087815255402977646, 0.087220877197025176, 0.086648297514724804, 0.085641447106788948, 0.085641447106788948, 0.085641447106788948, 0.085641447106788948, 0.085641447106788948, 0.085641447106788948, 0.084012864463609796, 0.083724882448298246, 0.083685930693360028, 0.082673071111457677, 0.082177335854618866, 0.081396192288234706, 0.07920884913725601, 0.078264936154609097, 0.074873918306017515, 0.074752523392866824, 0.063650204904313015, 0.056751208123973988, 0.053418302788745708, 0.044653435955511682, 0.044458994715454753, 0.040010508873757449, 0.037472631084346987, 0.0082022255608358886, 0.0072699922390287962, -0.0048018386994239104, -0.0070060323186668452, -0.010895880330213378, -0.020736033797986522, -0.028747243195871758, -0.030453321029221572, -0.043749821627929841, -0.054094425720053775, -0.063532212490196388, -0.068711177966265621, -0.087938076406901849, -0.096636212210379754]
feature_list_index = [37, 5, 1, 17, 49, 21, 85, 53, 33, 122, 112, 132, 65, 146, 152, 116, 136, 69, 142, 81, 126, 156, 96, 0, 166, 16, 150, 120, 32, 124, 134, 154, 162, 170, 144, 114, 160, 164, 130, 140, 97, 19, 3, 4, 20, 35, 51, 36, 7, 63, 23, 39, 83, 133, 123, 34, 113, 98, 2, 107, 79, 18, 67, 143, 71, 153, 15, 31, 87, 55, 47, 148, 158, 95, 118, 128, 13, 29, 66, 138, 168, 61, 163, 22, 77, 38, 84, 6, 82, 68, 99, 93, 45, 141, 50, 121, 135, 131, 111, 161, 125, 106, 102, 80, 115, 11, 59, 75, 108, 151, 52, 27, 129, 139, 43, 119, 110, 169, 103, 159, 165, 91, 137, 42, 127, 12, 28, 104, 117, 10, 109, 26, 44, 101, 9, 25, 41, 57, 73, 89, 30, 14, 105, 24, 8, 46, 70, 100, 90, 40, 86, 94, 149, 58, 155, 88, 64, 74, 60, 147, 56, 78, 72, 92, 157, 167, 48, 76, 145, 62, 54]
corrs = np.array(corrs)
# 0.9:5 0.85:11 0.8:21 0.75:27 0.7:37 0.65:40 0.6:47 0.55:63 0.5:68
# print corrs>0.9
# print len(corrs[corrs>0.5])
# gen_valid_values_2_DF(feature_list=feature_list)
# feature_list_index = feature_list_index[:68]
# gen_valid_values_2_DF(feature_list_index=feature_list_index)

# 成长值最小值为0.0006
# a = pd.read_csv('../data/SMPCUP2017_TrainingData_Task3.txt',delimiter='',index_col=0)
# a_values = a.values
# print a_values[a_values!=0].min()
# print len(a.columns)