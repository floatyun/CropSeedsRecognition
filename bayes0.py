import math
import csv
import random
 
class Bayes:
    def __init__(self,vectors,answers):
        self.vectors=vectors
        self.answers=answers
        # self.inputVector=inputVector
        # model_par以字典形式存放每一个类别的方差
        self.model_para={}
 
    def tarin_bayesModel(self):
        # 将训练集按照类别进行提取
        separated_class=self.separateByClass()
        # vectors是列表，包含的是每个类别对应的向量集
        for classValue, vectors in separated_class.items():
            # 将每一个类别的均值和方差保存在对应的键值对中
            self.model_para[classValue] = self.summarize(vectors)
        return self.model_para
 
    # 计算均值
    def mean(self,numbers):
        return sum(numbers) / float(len(numbers))
 
    # 计算方差，注意是分母是n-1
    def stdev(self,numbers):
        avg = self.mean(numbers)
        variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
        return math.sqrt(variance)
 
    # 对每一类样本的每个特征计算均值和方差，结果保存在列表中，依次为第一维特征、第二维特征等...的均值和方差
    def summarize(self,vectors):
        # zip利用 * 号操作符，可以将不同元组或者列表压缩为为列表集合。用来提取每类样本下的每一维的特征集合
        summaries = [(self.mean(attribute), self.stdev(attribute)) for attribute in zip(*vectors)]
        # 将代表类别的最后一个数据删掉，只保留均值和方差
        #del summaries[-1]
        return summaries
 
 
    # 将训练集按照类别进行提取，以字典形式存放，Key为类别，value为列表，列表中包含的是每个类别对应的向量集
    def separateByClass(self):
        #字典用于存放分类后的向量集合
        separated_class = {}
        for i in range(len(self.vectors)):
            vector = self.vectors[i]
            # vector[-1]为每组数据的类别
            if ((self.answers[i]-1) not in separated_class):
                separated_class[self.answers[i]-1] = []
            #     将每列数据存放在对应的类别下，列表形式
            separated_class[self.answers[i]-1].append(vector)
        return separated_class
 
    # 假定服从正态分布，对连续属性计算概率密度函数,公式参考周志华老师的西瓜书P151
    def calProbabilityDensity(self,x, mean, stdev):
        # x为待分类数据
        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
        return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
 
    # 计算待分类数据的联合概率
    def calClassProbabilities(self, inputVector):
        # summaries为训练好的贝叶斯模型参数, inputVector为待分类数据(单个)
        # probabilities用来保存待分类数据对每种类别的联合概率
        probabilities = {}
        # classValue为字典的key(类别) ,classSummaries为字典的vlaue(每个类别每维特征的均值和方差),列表形式
        for classValue, classSummaries in self.model_para.items():
            probabilities[classValue] = 1
            # len(classSummaries)表示有多少特征维度
            for i in range(len(classSummaries)):
                # mean, stdev分贝别表示每维特征对应的均值和方差
                mean, stdev = classSummaries[i]
                # 提取待分类数据的i维数据值
                x = inputVector[i]
                # 计算联合概率密度
                probabilities[classValue] *= self.calProbabilityDensity(x, mean, stdev)
        # 返回概率最大的类别
        prediction=max(probabilities,key=probabilities.get)
        return prediction+1
 
# # 准备数据
# def loadCsv(filename):
#     lines = csv.reader(open(filename, "r"))
#     dataset = list(lines)
#     for i in range(len(dataset)):
#         dataset[i] = [float(x) for x in dataset[i]]
#     return dataset
 
# # 将原始数据集划分为训练集和测试集，splitRatio为划分比例。
# def splitDataset(dataset, splitRatio):
#     trainSize = int(len(dataset) * splitRatio)
#     trainSet = []
#     copy = list(dataset)
#     while len(trainSet) < trainSize:
#         index = random.randrange(len(copy))
#         # 原始数据集剔除训练集之后剩下的就是测试集
#         trainSet.append(copy.pop(index))
#     return [trainSet, copy]
 
# # 计算分类准确率
# def calAccuracy(testData,bayes):
#     correct_nums=0
#     for i in range(len(testData)):
#         # 逐次计算每一个数据的分类类别
#         if  testData[i][-1]== bayes.calClassProbabilities(testData[i]):
#             correct_nums += 1
#     return correct_nums

bayes=Bayes([],[])

def train(vectors, answers):
    """
    种子类别
    :param vectors: 一个列表，每个vector[i]代表一个种子的向量
    :param answers: answers[i]代表vectors[i]种子类别（1，2，3）
    :return: 无返回值
    """
    global bayes
    bayes=Bayes(vectors, answers)
    bayes.tarin_bayesModel()

def recognize(vector):
    """
    识别种子类别
    :param vector: 要识别的种子的向量
    :return: 返回识别的种子类别的编号（1,2,3）
    """
    
    return bayes.calClassProbabilities(vector)


def main():
    filename = ''
 
    # 训练集和测试集的划分比例
    
 
    bayes=Bayes(trainData)
    # model为训练之后的bayes分类器模型的概率参数
    model=bayes.tarin_bayesModel()
    # print(model)
    correct_nums=calAccuracy(testData, bayes)
    print("分类准确率 %f%%"%(correct_nums/len(testData) * 100.0))
 
 
if __name__=="__main__":
    main()

