import time
import numpy as np
from numpy import random
import pandas as pd
from initial_import import import_dataset, import_training_set
from sklearn.preprocessing import KBinsDiscretizer


def compute_cond_entropy(data, name1, name2):
    prob_xy = data.groupby(name2)[name1].value_counts(sort=False)/data.groupby(name2)[name1].count()
    entropy = np.log2(prob_xy)*prob_xy
    entropy = entropy.groupby(name2).sum()
    entropy_xy = -(data[name2].value_counts(sort=False)/data[name2].count()*entropy).sum()
    return entropy_xy


def compute_entropy(serie):
    prob = serie.value_counts()/serie.count()
    return -(prob*np.log2(prob)).sum()


def compute_uncertainties(data, name1, name2):
    hx = compute_entropy(data[name1])
    hy = compute_entropy(data[name2])
    hxy = compute_cond_entropy(data, name1, name2)
    uncert = 2*(hx-hxy)/(hx+hy)
    return uncert


def next_element(data, element):
    """ """
    pos = None

    for i in range(data.shape[0]):
        if data.iloc[i, 1] == element[1]:
            pos = i

    if pos is None:
        return pos
    else:
        if data.iloc[-1, 1] == element[1]:
            return None
        else:
            next = data.iloc[pos+1, :]
            return next


def main(treshold, data=None):
    if data is None:
        data = import_training_set()  # import training set

    start = time.time()  # useful to compute computational time

    data = data.fillna(0)  # fill nan values with zero

    # create a vector containig features' names
    column_names = np.zeros(data.shape[1])
    column_names = data.columns
    no_change = ["date", "action", "ts_id", "feature_0"]

    # discretize dataset
    discr = KBinsDiscretizer(n_bins=1000, encode='ordinal', strategy='uniform')
    print("Discretiziong dataset")
    for name in column_names:
        if name not in no_change:
            print("Discretizing {}".format(name))
            data.loc[:, [name]] = discr.fit_transform(data.loc[:, [name]])

    # compute execution time discretization
    mins = (time.time()-start)//60
    sec = (time.time()-start) % 60
    print('Time to discretize dataset: {} min {:.2f} sec\n'.format(mins, sec))

    start_fcbf = time.time()
    SU = []  # empty list will contain SU values over treshold
    feat = []  # empty list will contain feature names with SU over treshold

    print("Applying FCBF algorithm")

    # start of FCBF algorithm
    # 1) create a dataframe containing SU values over treshold
    for name in column_names:
        x = compute_uncertainties(data, name, "action")
        if x > treshold and x != 1.:
            SU.append(x)
            feat.append(name)
    # trasform list into dataframe
    dict = {"Features": feat, "SU": SU}
    rel_feat = pd.DataFrame(dict)
    rel_feat = rel_feat.sort_values(by=["SU"], ascending=False)

    disc = len(column_names)-rel_feat.shape[0]  # number of features discarded
    print("We have discarded {} features".format(disc))
    print("The feature with a SU value above {} are".format(treshold))
    print(rel_feat)

    # select predominant features
    fp = rel_feat.iloc[0, :]
    while fp is not None:
        #print("fp is {}".format(fp[0]))
        fq = next_element(rel_feat, fp)
        if fq is not None:
            while fq is not None:
                fq1 = fq
                if compute_uncertainties(data, fp[0], fq[0]) > compute_uncertainties(data, fq[0], "action"):
                    print("{} is useless".format(fq[0]))
                    rel_feat = rel_feat[rel_feat != fq]
                    rel_feat = rel_feat.dropna()
                    fq = next_element(rel_feat, fq1)

                else:
                    fq = next_element(rel_feat, fq)
        fp = next_element(rel_feat, fp)

    # compute execution time FCBF
    mins = (time.time()-start_fcbf)//60
    sec = (time.time()-start_fcbf) % 60
    print('Time to compute FCBF algorithm: {} min {:.2f} sec\n'.format(mins, sec))

    return rel_feat
