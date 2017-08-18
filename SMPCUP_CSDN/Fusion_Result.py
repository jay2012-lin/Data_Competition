# -*- coding:utf-8 -*-
# 日期：2017/8/10 时间：下午3:18
# Author:Jaylin

def fusion_result():
    result_list = ['task1_result.txt','task2_result.txt','task3_result.txt']
    result_file = 'final.txt'
    file1_intro = 'contentid,keyword1,keyword2,keyword3'
    file2_intro = 'userid,interest1,interest2,interest3'
    file3_intro = 'userid,growthvaluee'

    with open(result_file,'wb') as fw:
        for no,file in enumerate(result_list,1):
            pro = '<task' + str(no) + '>'
            fw.write(pro + '\n')
            if no == 1:
                fw.write(file1_intro+'\n')
            if no == 2:
                fw.write(file2_intro + '\n')
            if no == 3:
                fw.write(file3_intro + '\n')

            with open(file,'rb') as fr:
                for line in fr:
                    fw.write(line)

            post = '</task' + str(no) + '>'
            fw.write(post + '\n')
            print '%s is finished!'%file
fusion_result()