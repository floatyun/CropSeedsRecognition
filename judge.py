from prepare_data import Data
import knn


def test(train,recognize):
    """
    测试并计算正确率
    :param train: 训练函数（参见knn的训练函数的样子）
    :param recognize: 识别函数（参见knn的识别的样子）
    :return: 无
    """
    data = Data()
    data.read_data()
    # 划分训练集
    data.divide_data()
    vectors,answers = data.get_train_set()
    train(vectors,answers)
    vectors, answers = data.get_test_set()
    test_count = len(vectors)
    right_count = 0
    class_right_count = dict()
    class_test_count = dict()
    for c in data.answer_set:
        class_right_count[c] = 0
        class_test_count[c] = 0
    for i in range(test_count):
        test_result = recognize(vectors[i])
        class_test_count[answers[i]] += 1
        if test_result == answers[i]:
            right_count += 1
            class_right_count[test_result] += 1
            print("对输入的种子向量" + str(vectors[i]) + "正确识别为类别"+ str(test_result))
        else:
            print("对输入的种子向量"+str(vectors[i])+"错误的识别为类别"
                  +str(test_result)+".而该种子实际上属于类别"+str(answers[i]))
    print("总正确率：%.1f%%" % (right_count/test_count * 100.0))
    for c in data.answer_set:
        print("类别"+str(c)+("的正确率为%.1f%%" % (class_right_count[c]/class_test_count[c] * 100.0)))


if __name__ == '__main__':
    test(knn.train, knn.recognize)
