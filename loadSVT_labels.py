import numpy as np

with open("SVT_labels.npy", 'rb') as l:
    labels = np.load(l, allow_pickle=True)
    # labels = data['labels']
    # print(labels)
    print(labels[0])