"""
本文件用于载入测试数据及将测试数据划分为训练集和测试集
"""
from random import randint


class Data:
    def __init__(self):
        self.filename="seeds_dataset.txt"
        self.all_vector = []
        self.all_answer = []
        self.is_train = []  # 记录是否是训练集
        # data_dict的key是类别种类编码，值是一个列表。
        # 列表的每个元素是一个下标（all_vectord的下标，同时也是all_answer的下标）。
        self.data_dict = dict()
        # 答案的集合，或者说标签的集合，只能是1,2,3
        self.answer_set = {1, 2, 3}
        for answer in self.answer_set:
            self.data_dict[answer] = [] # 初始化空的列表

    def read_data(self):
        all_data = []
        with open(self.filename) as f:
            all_data = f.readlines()
        i = 0
        for line in all_data:
            float_list = list(map(float, line.split()))
            vector = float_list[:-1]
            answer = int(float_list[-1])
            self.data_dict[answer].append(i)
            self.all_vector.append(vector)
            self.all_answer.append(answer)
            self.is_train.append(False)
            i += 1

    def get_m_lucky_index(self,n,m):
        """
        n个元素(0,1,2,..n-1)中等概率随机抽m个元素
        :param n: n个元素
        :param m: 需要抽取的元素个数
        :return: 幸运的m个下标的列表
        """
        lucky = []
        for i in range(n):
            if randint(0,n-i-1) < m:
                lucky.append(i)
                m -= 1
        return lucky

    def divide_data(self):
        """
        将数据的中的向量，每一类中将会随机一半变成训练集，其余变成测试集
        :return:
        """
        for i in range(len(self.is_train)):
            self.is_train[i] = False
        for a in self.answer_set:
            total = len(self.data_dict[a])  # 类别a的总数目
            train_count = total//2
            # print("total = ", total)
            # print("train_count=", train_count)
            train = self.get_m_lucky_index(total,train_count)
            # print("train = ", train)
            # print("total = ", total)
            # print("train_count=",train_count)
            for i in train:
                pos = self.data_dict[a][i]
                self.is_train[pos] = True

    def get_train_set(self):
        """
        获取训练集
        :return: vectors,answers
        vectors是训练集的种子向量的列表
        answers是相对应的答案
        """
        vectors = []
        answers = []
        for i in range(len(self.is_train)):
            if self.is_train[i]:
                vectors.append(self.all_vector[i])
                answers.append(self.all_answer[i])
        return vectors,answers

    def get_test_set(self):
        """
        获取测试集
        :return: vectors,answers
        vectors是测试集的种子向量的列表
        answers是相对应的答案
        """
        vectors = []
        answers = []
        for i in range(len(self.is_train)):
            if not self.is_train[i]:
                vectors.append(self.all_vector[i])
                answers.append(self.all_answer[i])
        return vectors, answers


if __name__ == '__main__':
    data = Data()
    data.read_data()
