from prepare_data import Data
import numpy as np
from mtr_to_img import *


def get_element(dis, results, n):
    """
    根据dis和results寻找类间距离最大的两个“类”(种子r,c)的编号
    :param dis: 类间距离矩阵
    :param results: 描述每个种子归属哪个类
    :param n: 初始的种子数
    :return: r,c
    """
    r = c = None
    for i in range(n):
        if results[i] != i:
            continue
        for j in range(i):
            if j != results[j]:
                continue
            if r is None:
                r, c = i, j
            elif dis[r, c] > dis[i, j]:
                r, c = i, j
    if r is None:
        raise ValueError("全部变成一类了，无法找到距离最大的两个类！")
    return r, c


def merge(dis, results, n, r, c):
    """
    合并两个类间距离最大的类r, c
    更新类间距离举证和results
    :param dis: 类间距离矩阵
    :param results: 描述每个种子归属哪个类
    :param n: 初始的种子数
    :param r: 要合并的类1
    :param c: 要合并的类2
    :return:
    """
    # 类编号选择较小的数
    # 保证r<c
    if r > c:
        r, c = r, c
    results[c] = r  # 让c类归属与r类
    # 更新距离矩阵
    # c类不存在了，故dis[,c]和dis[c,]不再有意义，不管即可
    # 更新dis[r,]与dis[,r]即可
    for i in range(n):
        if i != results[i] or i == r:
            continue
        t = max(dis[i, r], dis[i, c])
        dis[r, i] = dis[i, r] = t


def get_root(results, i):
    if i == results[i]:
        return i
    else:
        results[i] = get_root(results, results[i])
        return results[i]


def hierarchical_cluster(seeds:list, threshold=7.72):
    """
    层次聚类
    :param seeds: 输入的种子数组(每个种子都是一个list列表)
    :param threhold: 停止的聚类的距离阈值
    :return: results, classes
    一个列表group。group和vectors等长。group[i]表示vectors[i]的分类编号
    classes是一个列表，classes[i]是编号为ip的成员的列表
    """
    vectors = []
    for seed in seeds:
        vectors.append(np.array(seed))
    n = len(vectors)
    results = [i for i in range(n)]
    dis = np.zeros([n, n])
    # 求初始的距离矩阵dis
    for i in range(n):
        for j in range(i):
            dis[i, j] = dis[j, i] = np.linalg.norm(vectors[i] - vectors[j])
    print("mean = ", dis.mean())
    print("min = ", dis.min())
    print("max = ", dis.max())
    print("threshold = ", threshold)
    # 重复以下迭代过程直到最大距离已经达到了阈值
    # 1. 从距离矩阵中找类间距离最大的两个类
    # 2. 将这两个类合并
    merge_count = 0
    while True:
        i, j = get_element(dis, results, n)
        tmp = dis[i, j]
        if dis[i, j] >= threshold:
            break
        merge(dis, results, n, i, j)
        merge_count += 1
        # if merge_count%10 == 0:
        #     print("已经merge{}次了,这次merge之前的最大类间距离是{}".
        #           format(merge_count, tmp))
    class_index_dict = dict()
    j = 0
    for i in range(n):
        if i == results[i]:
            class_index_dict[i] = j
            j += 1
    classes = [[] for i in range(j)]
    for i in range(n):
        ri = get_root(results, i)
        classes[class_index_dict[ri]].append(i)
    return results, classes


def judge(real_classes, classes):
    """
    聚类评价
    :return: 输出聚类结果的构成 例如聚类成了三个类1,2,3输出
    聚类1=0.9真类1+0.2的真类2+0.1的真类3
    """
    mtr = np.zeros([len(classes), 3])
    for i in range(len(classes)):
        for j in classes[i]:
            mtr[i, real_classes[j]-1] += 1
    print("""
下面展示分类结果的情况。
行是聚类出来的一个类，列是表示真实的种子的一个类别。
值是个数。例如第一行第一列表示聚类出来的类1中实际上是1类种子的种子数。
分类结果展示：
""")
    print(mtr)
    a = mtr.sum(axis=0)
    print("种子真实分类各类的种子数目")
    print(a)
    mtr = mtr/a
    print("下面展示的是上面的矩阵转换为占比的情况")
    print(mtr)
    filename = input("请输入你要保存的矩阵图的文件名：\n")
    mtr_to_img(mtr, filename)


def main():
    data = Data()
    data.read_data()
    # 下面的参数用于找出比较好的阈值
    # l = 7
    # r = 11
    # step = 0.25
    # t = l
    # while t < r:
    #     print("T = ", t)
    #     results, classes = hierarchical_cluster(data.all_vector,threshold=t)
    #     judge(data.all_answer, classes)
    #     t += step
    #     os.system("pause")
    results, classes = hierarchical_cluster(data.all_vector)
    judge(data.all_answer, classes)


if __name__ == '__main__':
    main()
