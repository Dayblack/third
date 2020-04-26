#!/usr/bin/env python
# coding: utf-8

# In[3]:


#针对信用卡违约率的分析
"""数据集描述
数据集包括了 2013 年 9 月份两天时间内的信用卡交易数据，284807 笔交易中，一共有 492 笔是欺诈行为。输入数据一共包括了 28 个特征 V1，V2，……V28 对应的取值，以及交易时间 Time 和交易金额 Amount。可能为了保护数据隐私，不知道 V1 到 V28 这些特征代表的具体含义。
"""
#目标：构建一个信用卡欺诈分析的分类器，由数据描述可知，此分类数据不平衡。


# In[4]:


#项目流程
"""
1.加载数据
2.探索数据，用数据可视化的方式查看分类结果的情况，以及随着时间的变化，欺诈交易和正常
交易的分布情况。源数据集中，V1-V28 的特征值都经过 PCA 的变换，但是其余的两个字段，Ti
me 和 Amount 还需要进行规范化。Time 字段和交易本身是否为欺诈交易无关，因此不作为特征
选择，只需要对 Amount 做数据规范化。
3.使用Logistic回归。精确率、召回率和 F1 值。同时将精确率-召回率进行了可视化呈现。
"""


# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
#导入逻辑回归
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#导入指标：混淆矩阵、精确率-召回率曲线
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# In[6]:


# 混淆矩阵可视化
def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion matrix"', cmap = plt.cm.Blues) :
    plt.figure()
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 0)
    plt.yticks(tick_marks, classes)
 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])) :
        plt.text(j, i, cm[i, j],
                 horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')
 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# In[7]:


# 显示模型评估结果
def show_metrics():
    tp = cm[1,1]
    fn = cm[1,0]
    fp = cm[0,1]
    tn = cm[0,0]
    print('精确率: {:.3f}'.format(tp/(tp+fp)))
    print('召回率: {:.3f}'.format(tp/(tp+fn)))
    print('F1值: {:.3f}'.format(2*(((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn))))))


# In[8]:


# 绘制精确率-召回率曲线
def plot_precision_recall():
    plt.step(recall, precision, color = 'b', alpha = 0.2, where = 'post')
    plt.fill_between(recall, precision, step ='post', alpha = 0.2, color = 'b')
    plt.plot(recall, precision, linewidth=2)
    plt.xlim([0.0,1])
    plt.ylim([0.0,1.05])
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('精确率-召回率 曲线')
    plt.show();


# In[9]:


# 数据加载
data = pd.read_csv(r'C:\Users\ASUS\Desktop\data\creditcard.csv')
# 数据探索
print(data.describe())
# 设置plt正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
# 绘制类别分布
plt.figure()
ax = sns.countplot(x = 'Class', data = data)
plt.title('类别分布')
plt.show()


# In[10]:


# 显示交易笔数，欺诈交易笔数
num = len(data)
num_fraud = len(data[data['Class']==1]) 
print('总交易笔数: ', num)
print('诈骗交易笔数：', num_fraud)
print('诈骗交易比例：{:.6f}'.format(num_fraud/num))


# In[11]:


# 欺诈和正常交易可视化
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15,8))
bins = 50
ax1.hist(data.Time[data.Class == 1], bins = bins, color = 'deeppink')
ax1.set_title('诈骗交易')
ax2.hist(data.Time[data.Class == 0], bins = bins, color = 'deepskyblue')
ax2.set_title('正常交易')
plt.xlabel('时间')
plt.ylabel('交易次数')
plt.show()


# In[32]:


# 对Amount进行数据规范化
# 调用fit_transform方法，两步并作一步，根据已有的数据fit创建一个标准化转换器并使用transform去转换训练数据
# data['Amount_Norm'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
# 特征选择
# y = np.array(data.Class.tolist())
# data = data.drop(['Time','Amount','Class'],axis=1)
print(type(data))
X = np.array(data.as_matrix())
print(X)
# 准备训练集和测试集
train_x, test_x, train_y, test_y = train_test_split (X, y, test_size = 0.1, random_state = 33)


# In[33]:


# 逻辑回归分类
clf = LogisticRegression()
clf.fit(train_x, train_y)
predict_y = clf.predict(test_x)
# 预测样本的置信分数
score_y = clf.decision_function(test_x)  
# 计算混淆矩阵，并显示
cm = confusion_matrix(test_y, predict_y)
class_names = [0,1]
# 显示混淆矩阵
plot_confusion_matrix(cm, classes = class_names, title = '逻辑回归 混淆矩阵')
# 显示模型评估分数
show_metrics()
# 计算精确率，召回率，阈值用于可视化
precision, recall, thresholds = precision_recall_curve(test_y, score_y)
plot_precision_recall()


# In[35]:


#线性SVM分类
from sklearn import svm
model=svm.LinearSVC()
model.fit(train_x,train_y)
predict_y=model.predict(test_x)
# 预测样本的置信分数
score_y = model.decision_function(test_x)  
# 计算混淆矩阵，并显示
cm = confusion_matrix(test_y, predict_y)
class_names = [0,1]
# 显示混淆矩阵
plot_confusion_matrix(cm, classes = class_names, title = 'SVM 混淆矩阵')
# 显示模型评估分数
show_metrics()
# 计算精确率，召回率，阈值用于可视化
precision, recall, thresholds = precision_recall_curve(test_y, score_y)
plot_precision_recall()


# In[43]:


# 显示特征向量的重要程度
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
# 显示特征向量的重要程度
coeffs = clf.coef_
df_co = pd.DataFrame(coeffs.T, columns=["importance_"])
# 下标设置为Feature Name
df_co.index = data.columns
df_co.sort_values("importance_", ascending=True, inplace=True)
df_co.importance_.plot(kind="barh")
plt.title("Feature Importance")
plt.show()


# In[51]:


y


# In[ ]:




