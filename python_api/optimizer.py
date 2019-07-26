import numpy as np
import copy
import random


class GAOptimizer:
    def __init__(self, length, pc, pm, iniPop):
        self.length = length
        self.pc = pc
        self.pm = pm
        if iniPop is not None:
            self.popNow = iniPop
            self.size = len(iniPop)
        else:
            print("需要输入初始群体")
            assert 0

    def getNextGeneration(self, currentScores):
        self.popNow = self._choose(currentScores)
        np.random.shuffle(self.popNow)
        self._cross()
        self._variation()
        return self.popNow

    def _choose(self, currentScores):
        scoreSum = np.sum(currentScores)
        if scoreSum == 0:
            probs = np.array([0 for i in range(len(currentScores))])
        else:
            probs = np.array([currentScores[i]/scoreSum for i in range(len(currentScores))])
        chooseNums = np.array([int(prob*self.size) for prob in probs])
        moreToChooseNum = self.size-np.sum(chooseNums)
        leftProbs = np.array([probs[i]*self.size-chooseNums[i] for i in range(len(currentScores))])
        biggestIndexs = np.argpartition(leftProbs, -moreToChooseNum)[-moreToChooseNum:]
        for index in biggestIndexs:
            chooseNums[index] += 1
        result = []
        for i in range(len(chooseNums)):
            num = chooseNums[i]
            for j in range(num):
                result.append(copy.deepcopy(self.popNow[i]))
        return result

    def _cross(self):
        before = None
        for i in range(self.size):
            prob = random.random()
            if prob < self.pc:
                if before is None:
                    before = self.popNow[i]
                else:
                    crossPosition = random.randint(1, self.length-1)
                    for j in range(crossPosition, self.length):
                        tmp = before[j]
                        before[j] = self.popNow[i][j]
                        self.popNow[i][j] = tmp
                    before = None

    def _variation(self):
        for i in range(self.size):
            prob = random.random()
            if prob < self.pm:
                variationPosition = random.randint(0, self.length-1)
                option = random.random()
                if option<0.5:
                    self.popNow[i][variationPosition] = self.popNow[i][variationPosition] * 1.2
                else:
                    self.popNow[i][variationPosition] = self.popNow[i][variationPosition] / 1.2
    