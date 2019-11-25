from prepare_data import Data
import knn
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def test(train,recognize,k,data):
    """
    测试并计算正确率
    :param train: 训练函数（参见knn的训练函数的样子）
    :param recognize: 识别函数（参见knn的识别的样子）
    :return: 无
    """
    vectors,answers = data.get_train_set()
    train(vectors,answers)
    vectors, answers = data.get_test_set()
    test_count = len(vectors)
    right_count = 0
    predict = []
    # mtr[i][j]表示本来是类别i，但是识别成类别b的比例
    for i in range(test_count):
        test_result = recognize(vectors[i])
        predict.append(test_result)
        if test_result == answers[i]:
            right_count += 1
            # print("对输入的种子向量" + str(vectors[i]) + "正确识别为类别"+ str(test_result))
        else:
            pass
            # print("对输入的种子向量"+str(vectors[i])+"错误的识别为类别"
            #       +str(test_result)+".而该种子实际上属于类别"+str(answers[i]))
    print("总正确率：%.1f%%" % (right_count/test_count * 100.0))
    # # 展示混淆矩阵
    # sns.set()
    # f, ax = plt.subplots()
    # mtr = confusion_matrix(answers, predict, labels=[1,2,3])
    # print(mtr)  # 打印出来看看
    # sns.heatmap(mtr, annot=True, ax=ax)  # 画热力图
    # ax.set_title('混淆矩阵')  # 标题
    # ax.set_xlabel('预测结果')  # x轴
    # ax.set_ylabel('实际分类')  # y轴


if __name__ == '__main__':
    data = Data()
    data.read_data()
    # 划分训练集
    data.divide_data()
    for k in range(5,50):
        print("k = ",k,end="\t")
        test(knn.train, knn.recognize,data=data, k=k)
