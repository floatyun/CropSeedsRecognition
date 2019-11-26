from prepare_data import Data
import knn
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image


def test(clear,train,recognize,data, dispaly_process=False):
    """
    测试并计算正确率
    :param train: 训练函数（参见knn的训练函数的样子）
    :param recognize: 识别函数（参见knn的识别的样子）
    :return: 无
    """
    clear()
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
            if dispaly_process:
                print("对输入的种子向量" + str(vectors[i]) + "正确识别为类别"+ str(test_result))
        else:
            if dispaly_process:
                print("对输入的种子向量"+str(vectors[i])+"错误的识别为类别"
                      +str(test_result)+".而该种子实际上属于类别"+str(answers[i]))
    print("总正确率：%.1f%%" % (right_count/test_count * 100.0))
    return right_count/test_count,answers,predict


def get_head_img(answers,predict,labels,filename="heat_img.jpg"):
    sns.set()
    f, ax = plt.subplots()
    mtr = confusion_matrix(answers, predict, labels=labels)
    print(mtr)  # 打印出来看看
    sns.heatmap(mtr, annot=True, ax=ax)  # 画热力图
    ax.set_title('confusion matrix')  # 标题
    ax.set_xlabel('predict answer')  # x轴
    ax.set_ylabel('real answer')  # y轴
    f.savefig(filename, bbox_inches='tight')


def show_heat_img(filename):
    img = Image.open(filename)
    plt.figure("Image")  # 图像窗口名称
    plt.imshow(img)
    plt.axis('on')  # 关掉坐标轴为 off
    plt.title('image')  # 图像题目
    plt.show()



if __name__ == '__main__':
    data = Data()
    print("读取数据.......")
    data.read_data()
    print("数据读取完毕。")
    # 划分训练集
    sum = 0
    predict = []
    answers = []
    test_count = int(input("测试knn算法，请输入测试次数：\n"))
    for i in range(test_count):
        data.divide_data()
        rate,r,p = test(knn.clear,knn.train, knn.recognize,data=data)
        sum += rate
        predict += p
        answers += r
    sum /= test_count
    print("一共重复测试了"+str(test_count)+"次")
    print("每次测试了"+str(len(answers)//test_count)+"个种子。")
    print("平均正确率：%.1f%%" % (sum * 100.0))
    filename = "knn_heat_img.jpg"
    get_head_img(answers,predict,[1,2,3],filename=filename)
    show_heat_img(filename)

