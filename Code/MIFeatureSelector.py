import numpy as np
import pandas as pd


class CategoricalMI:

    def __init__(self, data, target_name, feature_names):

        self.y = data[target_name]
        self.feature_names = feature_names
        self.X = data[feature_names]
        self.X = self.X.astype(str)
        self.y = self.y.astype(str)
        self.nrows = len(self.X)

    def joint_entropy(self, x, y, p):
        sum = 0
        for x in range(x):
            for y in range(y):
                if p[x][y] == 0:

                    continue
                sum += -(p[x][y] * np.log2((p[x][y])))

        return sum

    def entropy(self, x):

        sum = 0
        for i in x:
            sum -= i * np.log2(i)
        return sum

    def creat_joint_pmt(self, fname):

        ct = pd.crosstab(self.y, self.X[fname])
        jp = np.array(ct)
        jp = jp / len(self.nrows)

        return jp

    def run(self):

        self.mis = {}
        mi_scores = []

        for fname in self.feature_names:

            jpmf = self.creat_joint_pmt(fname)
            h_x_y = self.joint_entropy(jpmf.shape[0], jpmf.shape[1], jpmf)

            xpmf = jpmf.sum(axis=0)
            ypmf = jpmf.sum(axis=1)

            h_x = self.entropy(xpmf)
            h_y = self.entropy(ypmf)

            cond_x_y = h_x_y - h_y
            cond_y_x = h_x_y - h_x

            i_x_y = h_x - cond_x_y

            mi_scores.append(i_x_y)
        
        mi_scores = sorted(zip(mi_scores, self.feature_names), key=lambda x : x[0])
        self.mis = {fn: mi for mi, fn in mi_scores}

    def report(self):

        for fname, mi in self.mis.items():
            
            print(f"{fname} mutual information with target : {mi}")
