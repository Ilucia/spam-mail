# -*- coding: utf-8 -*-
# @Author: niu
# @Date:   2019-08-22 16:34:21
# @Last Modified by:   niu
# @Last Modified time: 2019-08-22 18:00:30
import os  # 用于文件操作
import collections  # 用于统计操作
import numpy as np  # 用于对二维列表的操作，导包中使用as是改变模块的引用对象名字
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import pickle


def make_Dictionary(tran_dir):
    # 获取到参数文件夹下的所有文件名
    emails = [os.path.join(tran_dir, f) for f in os.listdir(tran_dir)]
    # 定义一个Counter类，存储单词和出现次数
    dictionary = collections.Counter()
    # 定义一个空列表,用于存储所有的单词
    all_words = []
    for mail in emails:
        with open(mail) as m:
            for i, line in enumerate(m):
                if i == 2:
                    words = line.split()
                    # 将字符以空格分割之后，保存在all_words列表中
                    all_words += words
                    # 返回一个字典，包含了all_words中的所有对象以及对应的数量
                    # dictionary.update(collections.Counter(all_words))
    dictionary = collections.Counter(all_words)
    # 获取到字典中的所有主键，放在list_to_remove列表中
    list_to_remove = dictionary.keys()
    # print(type(all_words))
    # print(type(list_to_remove))
    # python2中keys()返回的列表，Python3中返回的属性是dict_keys,所以list转换一下
    for item in list(list_to_remove):
        if item.isalpha() == False:
            del dictionary[item]
        # 使用字符串len()方法，判断主键是一个字符时，从字典中删除
        elif len(item) == 1:
            del dictionary[item]
    # 返回一个排序后的列表。从次数高到次数低。如果n没有被指定，则返回所有元素。当多个元素计数值相同时，排列是无确定顺序的。
    dictionary = dictionary.most_common(3000)
    # print(all_words)
    # print(dictionary)
    # 返回字典，即此方法是返回文件夹查找之后的单词，次数的字典
    return dictionary


train_dir = 'train-mails'

# 读取保存的训练生成的词典
with open('dictionary.pkl', 'rb') as df:
    dictionary = pickle.load(df)

# 生成词典
# dictionary = make_Dictionary(train_dir)


# 将训练生成的词典保存到本地
# with open('dictionary.pkl', 'wb') as df:
#     pickle.dump(dictionary, df)


# 此方法用于生成一个特征向量矩阵，即将文件夹路径传入
# 生成，行为文件，列为词典中的3000个单词，这样一个二维列表
# [ij]表示第i个文件中，第j个单词出现的次数
# def extract_features_from_file(mail_dir):
#     # 构建一个文件列表
#     files = [os.path.join(mail_dir, fi) for fi in os.listdir(mail_dir)]
#     # zeros返回一个二维数组，行数是文件数量，列数是3000，代表词典中前3000个常用词，默认填充为0
#     features_matrix = np.zeros((len(files), 3000))
#     # 定义一个行标，i
#     docID = 0
#     for fil in files:
#         # 从文件列表遍历打开文件
#         with open(fil) as fi:
#             for i, line in enumerate(fi):
#                 # 文件第二行开始是正文
#                 if i == 2:
#                     words = line.split()
#                     # 将正文以空格分割，返回由分割后词组成的列表
#                     for word in words:
#                         # 遍历每个词，并定义一个列表，j
#                         wordID = 0
#                         # 通过变量i递增，enumerate返回一个enumerate对象，在字典中返回键值组成的索引序列。
#                         for i, d in enumerate(dictionary):
#                             # d[0]表示键值
#                             if d[0] == word:
#                                 # 将索引值赋给列标
#                                 wordID = i
#                                 # 正文中的每一个词进行计数，出现的次数便是，二维列表的[i,j]出值
#                                 features_matrix[docID,
#                                                 wordID] = words.count(word)
#         docID = docID + 1
#     # 返回对文件夹中文件的分析列表
#     return features_matrix


def extract_features_from_str(mail_text):
    words = mail_text.split()
    features_matrix = np.zeros((1, 3000))
    # 将正文以空格分割，返回由分割后词组成的列表
    for word in words:
        # 遍历每个词，并定义一个列表，j
        wordID = 0
        # 通过变量i递增，enumerate返回一个enumerate对象，在字典中返回键值组成的索引序列。
        for i, d in enumerate(dictionary):
            # d[0]表示键值
            if d[0] == word:
                # 将索引值赋给列标
                wordID = i
                # 正文中的每一个词进行计数，出现的次数便是，二维列表的[i,j]出值
                features_matrix[0, wordID] = words.count(word)
    # 返回对文件夹中文件的分析列表
    return features_matrix


# 返回一个由0填充，长度为702的列表
train_labels = np.zeros(702)
# 对于初始来说，正常邮件和垃圾邮件的概率为1/2，所以将后半列表定义为1
train_labels[351:701] = 1
# 生成一个由训练文件而产生的分析表
# 这是为后面的学习构建一个模型
# train_matrix = extract_features_from_file(train_dir)

# 读取保存的训练结果分析表
with open('train_matrix.pkl', 'rb') as mf:
    train_matrix = pickle.load(mf)

# 保存生成的训练结果分析表
# with open('train_matrix.pkl', 'wb') as mf:
#     pickle.dump(train_matrix, mf)

# 训练SVM和贝叶斯模式
# 朴素贝叶斯分类器
# 条件概率P(A|B)，表示事件B发生的条件下，事件A发生的概率
# 全概率公式，P(A)=∑[P(Bi)P(A|Bi)],表示在所有因素Bi的影响下，事件A发生的概率
# 贝叶斯公式，P(Bi|A),表示事件A发生时，由因素Bi引发的概率

# 支持向量机
# 通常希望一个分类的过程是一个机器学习的过程。对于一些数据点，存在n维空间中，我们希望有一个n-1维的超平面去将数据点分为两类
# 或许存在不止一个超平面能够将数据按照要求分为两类，但是还是希望能找到分类最佳的平面，即使得两个不同类的数据点间隔最大的那个超平面
# 支持向量机将向量（数据点），映射到一个更高维的空间中，并在这个空间里建立一个最大间隔超平面
# 在分开数据的超平面的两侧建立两个相互平行的超平面，这两个平行平面的间隔越大，分类器的总误差越小
# SVM的关键在于核函数。低维空间向量集通常难于划分，解决的方法是将它们映射到高维空间。
# 但这个办法带来的困难就是计算复杂度的增加，而核函数正好巧妙地解决了这个问题。
# 也就是说，只要选用适当的核函数，就可以得到高维空间的分类函数。在SVM理论中，采用不同的核函数将导致不同的SVM算法。

model1 = MultinomialNB()
model2 = LinearSVC()

# fit函数表示训练某种模型。函数返回值一般为调用fit方法的对象本身

model1.fit(train_matrix, train_labels)
model2.fit(train_matrix, train_labels)


# 根据所训练的模型，进行预测


def predict_mail_is_spam(mail_text):
    test_matrix = extract_features_from_str(mail_text)
    result = model1.predict(test_matrix) + model2.predict(test_matrix)
    if result / 2 > 0.6:
        return True
    else:
        return False


# test_labels = np.zeros(1)
# test_labels[130:260] = 1


# confusion_matrix混淆矩阵，用于呈现算法性能的可视化效果
# 通过对角来判定准确率，其余为偏差量
# print(confusion_matrix(test_labels, result1))
print(predict_mail_is_spam("Nice to meet you!"))
print(predict_mail_is_spam("Subject: re : 5 . 939 binary comparison sure doerfer never binary comparison allow ( e . g . , turkic mongolic turkic , mongolic , ( manchu - ) tungusic once ) . contary , over over again binary comparison prove existence genetic relationship whole group . marcel erdal"
                           ))
print(predict_mail_is_spam("Subject: e-gift certificate # 212-6587900 - 82936699's our pleasure send gift certificate passion shoppe apply toward purchase item our online catalogue . automatic e-mail notification inform e-gift certificate purchase . generous person gift list below . don ' t delete message ! ' ll need claim code below place order . happy shop ! friend passion shoppe . * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * amount : $ 20 . 0 : secret admirer gift message : stuff news 'd kick . supposedly , really work . . . can't wait hear ! claim code bdjb-dg5m52 - 4pl4 order # 212-6587900 - 8293668 expiration date 15 - jun-99 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * using gift certificate easy : 1 . visit passion shoppe web site . 2 . select item want . 3 . select item want , hit order button . redeem gift certificate enter claim code order form . claim e-gift certificate , visit passion shoppe 's web site below . protection , site e-commerce secure encrypt order online : http : / / freehosting2 . . webjump . com / ku / kuretake-scientist / index . html * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * fine print : gift certificate must redeem passion shoppe web site . gift certificate redeemable cash . gift certificate unus portion gift certificate expire date list e-gift certificate earliest date permit under applicable law , whichever occur later . unus balance place gift certificate account . order exceed amount gift certificate , must pay balance credit card check tps 's e-commerce secure web site mail money order . * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * legal stuff fund unclaim gift certificate become property tps . wish receive reminder notification gift certificate name hold tps 's gift certificate account , remove name future reminder mailing enter enter name below . off221 @ excite . com although e-mail request update automatically , ca wa resident voicemail 888-294 - 238 . voice mail request check update once per month . * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *"))
