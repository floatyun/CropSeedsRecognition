class TR:
    def __init__(self,a=[],i=-1):
        self.vector=a
        self.answer=i

class RE:
    def __init__(self,a=0.0,i=-1):
        self.distance=a
        self.answer=i

    def __lt__(self,other):
        return self.distance<other.distance

def e_distance(x,y):
    nx = len(x)
    a=0
    for i in range(nx):
        a+=(x[i]-y[i])**2

    return a**0.5


trainingset=[]


def recognize(vector,k=15):
    """
    识别种子类别
    :param vector: 要识别的种子的向量
    :return: 返回识别的种子类别的编号（1,2,3）
    """
    num=[0 for i in range(3)]
    
    testset=[]
    for i in range(len(trainingset)):
        distance=e_distance(vector,trainingset[i].vector)
        testset.append(RE(distance,trainingset[i].answer))
    testset.sort()
    #print(testset)
    for i in range(k):
        num[testset[i].answer-1]+=1
    
    maxnum=0
    for i in range(3):
        if(num[i]>num[maxnum]):
            maxnum=i
    return maxnum+1


def train(vectors, answers):
    """
    种子类别
    :param vectors: 一个列表，每个vector[i]代表一个种子的向量
    :param answers: answers[i]代表vectors[i]种子类别（1，2，3）
    :return: 无返回值
    """
    num=[0 for i in range(3)]
    for i in range(len(vectors)):
        if(num[answers[i]-1]<50):
            num[answers[i]-1]+=1
            trainingset.append(TR(vectors[i],answers[i]))


if __name__=='__main__':
    a=[0 for i in range(2)]
    b=[1 for i in range(2)]

    print(e_distance(a,b))

