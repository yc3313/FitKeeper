import numpy as np
from sklearn.cluster import KMeans

def kmeans(db):
    # key is user_id, value is its group_id
    group = {}
    # key is group_id, value is its user_id
    get_group_member = {}

    data_list = []
    id = []
    for data in db:
        id.append(data[0])
        data_list.append(data[1:])

    df = np.array(data_list)
    k = 2

    for i in range(0, k):
        get_group_member[str(i)] = []
    clf = KMeans(n_clusters=k).fit(df)
    labels = clf.labels_
    for i in range(0, len(labels)):
        group[id[i]] = labels[i]
        get_group_member[str(labels[i])].append(id[i])

    return group, get_group_member
