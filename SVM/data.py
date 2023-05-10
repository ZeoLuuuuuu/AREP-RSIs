import numpy as np
import scipy.misc as im
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

def get_mstar_data(stage, width=128, height=128, crop_size=128, aug=False):
    data_dir = "MSTAR-10/train/" if stage == "train" else "MSTAR-10/test/" if stage == "test" else None
    print("------ " + stage + " ------")
    sub_dir = ["2S1", "BMP2", "BRDM_2", "BTR60", "BTR70", "D7", "T62", "T72", "ZIL131", "ZSU_23_4"]
    X = []
    y = []

    for i in range(len(sub_dir)):
        tmp_dir = data_dir + sub_dir[i] + "/"
        img_idx = [x for x in os.listdir(tmp_dir) if x.endswith(".jpeg")]
        print(sub_dir[i], len(img_idx))
        y += [i] * len(img_idx)
        for j in range(len(img_idx)):

           # img = im.imresize(im.imread((tmp_dir + img_idx[j])), [height, width])
           img=im.imread((tmp_dir + img_idx[j]))
           #img = img[(height - crop_size) // 2 : height - (height - crop_size) // 2, \
           #       (width - crop_size) // 2: width - (width - crop_size) // 2]
            # img = img[16:112, 16:112]   # crop
           X.append(img)

    return np.asarray(X), np.asarray(y)

def data_shuffle(X, y, seed=0):
    data = np.hstack([X, y[:, np.newaxis]])
    np.random.shuffle(data)
    return data[:, :-1], data[:, -1]

def one_hot(y_train, y_test):
    one_hot_trans = OneHotEncoder().fit(y_train[:, np.newaxis])
    return one_hot_trans.transform(y_train[:, np.newaxis]).toarray(), one_hot_trans.transform(y_test[:, np.newaxis]).toarray()

def mean_wise(X):
    return (X.T - np.mean(X, axis=1)).T

def pca(X_train, X_test, n):
    pca_trans = PCA(n_components=n).fit(X_train)
    return pca_trans.transform(X_train), pca_trans.transform(X_test)

"""
AttributeError: module 'scipy.misc' has no attribute 'imresize'
给予参考的配置：
scipy ==1.2.1;
Pillow ==6.0.0;

#利用 scipy 进行resize() 
new_image = scipy.misc.imresize(old_image, 0.99999, interp = 'cubic')
#利用 Image;
im = Image.fromarray(old_image)
size = tuple((np.array(im.size) * 0.99999).astype(int))
new_image = np.array(im.resize(size, PIL.Image.BICUBIC))
"""