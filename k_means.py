from prepare_data import Data
import numpy as np
import hierarchical_clustering as hc


def get_nearest_index(centers, p):
    """
    返回离点p最近的中心点centers[i]的下标i
    :param centers: 中心点的数组。
    :param param: 点p
    :return: i
    """
    x, dis = None, None
    for i in range(len(centers)):
        d = np.linalg.norm(centers[i] - p)
        if x is None or dis > d:
            x = i
            dis = d
    assert(x is not None)
    return x


def k_means_cluster(k, now_centers, seeds, enough_less_dis=1e-8, max_iterator_count=1e6):
    """
    kj均值聚类算法
    :param k: k
    :param now_centers: 初始的k个类中心
    :param seeds: 种子的向量数组（list）
    :param enough_less_dis: 足够小的距离阈值，距离小于该阈值认为点重合
    :param max_iterator_count: 最大迭代次数
    :return: centers, group, classes
    group是一个列表。group[i]表示seeds[i]的分类编号（从0开始）。
    classes是列表，classes[i]是存储了所有类别i的种子的下标。
    """
    # 根据最小距离原则将点划分到k个聚类中心所代表的类中
    # 对划分出来的k个类，计算类的“理想”中心（均值向量）
    # 如果现实和“理想”全部重合了，那么停止迭代
    # 由于都是实数，所以是否“重合”不能直接各个维度直接比较。
    # 本程序采用欧式距离度量，当小于一定的阈值即认为重合
    # 同时采取最大迭代次数
    vectors = []
    for seed in seeds:
        vectors.append(np.array(seed))
    n = len(vectors)
    m = len(seeds[0])  # 种子向量的维度数目
    results = [0] * n
    iterator_count = 0
    is_same = False
    while iterator_count < max_iterator_count:
        # 计算每个点所属的类别
        for i in range(n):
            results[i] = get_nearest_index(now_centers, vectors[i])
        # 求类中的向量和及向量个数
        next_centers = [np.zeros(m) for i in range(k)]
        count = [0] * k
        for i in range(n):
            next_centers[results[i]] += vectors[i]
            count[results[i]] += 1
        # 计算新的聚类中心并判断是否重合
        is_same = True
        for i in range(k):
            next_centers[i] /= count[i]  # 理论上应该不至于划分出来某个类一个点都没不可能发生除0错误
            d = np.linalg.norm(now_centers[i] - next_centers[i])
            if d >= enough_less_dis:
                is_same = False
        if is_same:
            break
        now_centers = next_centers
        iterator_count += 1
    if is_same:
        print("迭代次数是"+str(iterator_count))
        print("新的类中心已经和旧的类中心重合了")
    else:
        print("已经迭代了"+str(max_iterator_count)+"次了，但依旧没有达到新的类中心和旧的类中心重合的情况。")
    classes = [[] for i in range(k)]
    for i in range(n):
        classes[results[i]].append(i)
    return now_centers, results, classes


def main():
    data = Data()
    data.read_data()
    k = 3
    # 选取k个点作为初始的k个类的中心
    # init_center_indexes = data.get_m_lucky_index(len(data.all_answer), k)
    # init_center_indexes = [30+i*70 for i in range(k)]
    init_center_indexes = [i for i in range(k)]
    # 获取k个中心点的向量
    init_centers = [np.array(data.all_vector[i]) for i in init_center_indexes]

    centers, results, classes = \
        k_means_cluster(
            k, init_centers,data.all_vector)
    # print("最终确定的"+str(k)+"聚类中心如下所示：")
    # print(centers)
    hc.judge(data.all_answer, classes)


if __name__ == '__main__':
    main()
