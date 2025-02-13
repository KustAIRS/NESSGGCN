import numpy as np
import scipy.io as sio
import spectral as spy
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA


class DataReader():
    def __init__(self):
        self.data_cube = None
        self.g_truth = None

    @property
    def cube(self):
        return self.data_cube

    @property
    def truth(self):
        return self.g_truth.astype(np.int64)

    @property
    def normal_cube(self):
        return (self.data_cube - np.min(self.data_cube)) / (np.max(self.data_cube) - np.min(self.data_cube))

class WHU_Hi_HongHu(DataReader):
    def __init__(self):
        super(WHU_Hi_HongHu, self).__init__()
        raw_data_package = sio.loadmat(r"your path/WHU_Hi_HongHu.mat")
        self.data_cube = raw_data_package["WHU_Hi_HongHu"].astype(np.float32)
        truth = sio.loadmat("your path/WHU_Hi_HongHu_gt.mat")
        self.g_truth = truth["WHU_Hi_HongHu_gt"].astype(np.float32)





# PCA
def apply_PCA(data, num_components=75):
    new_data = np.reshape(data, (-1, data.shape[2]))
    pca = PCA(n_components=num_components, whiten=True)
    new_data = pca.fit_transform(new_data)
    new_data = np.reshape(new_data, (data.shape[0], data.shape[1], num_components))  
    return new_data, pca

def data_info(train_label=None, test_label=None, start=1):  
    class_num = np.max(train_label.astype('int32'))  
    if train_label is not None and test_label is not None:

        total_train_pixel = 0
        total_test_pixel = 0
        train_mat_num = Counter(train_label.flatten())
        test_mat_num = Counter(test_label.flatten())  

        for i in range(start, class_num + 1):
            print("class", i, "\t", train_mat_num[i], "\t", test_mat_num[i])
            total_train_pixel += train_mat_num[i]
            total_test_pixel += test_mat_num[i]
        print("total", "    \t", total_train_pixel, "\t", total_test_pixel)

    elif train_label is not None:
        total_pixel = 0
        data_mat_num = Counter(train_label.flatten())

        for i in range(start, class_num + 1):
            print("class", i, "\t", data_mat_num[i])
            total_pixel += data_mat_num[i]
        print("total:   ", total_pixel)

    else:
        raise ValueError("labels are None")
def list_to_colormap1(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.  
        if item == 1:
            y[index] = np.array([176,48,96]) / 255. 
        if item == 2:
            y[index] = np.array([0,255,255]) / 255. 
        if item == 3:
            y[index] = np.array([255,0,254]) / 255.  
        if item == 4:
            y[index] = np.array([160,32,239]) / 255.  
        if item == 5:
            y[index] = np.array([126,255,212]) / 255. 
        if item == 6:
            y[index] = np.array([128,255,0]) / 255. 
        if item == 7:
            y[index] = np.array([0,205,0]) / 255.  
        if item == 8:
            y[index] = np.array([0,255,1]) / 255. 
        if item == 9:
            y[index] = np.array([1,139,0]) / 255.  
        if item == 10:
            y[index] = np.array([254,0,0]) / 255.  
        if item == 11:
            y[index] = np.array([215,191,215]) / 255.  
        if item == 12:
            y[index] = np.array([255,127,80]) / 255. 
        if item == 13:
            y[index] = np.array([160,82,44]) / 255.  
        if item == 14:
            y[index] = np.array([255,255,255]) / 255. 
        if item == 15:
            y[index] = np.array([218,112,213]) / 255.  
        if item == 16:
            y[index] = np.array([0,0,254]) / 255.  
        if item == -1:
            y[index] = np.array([0, 0, 0]) / 255. 
    return y

def list_to_colormap2(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.  
        if item == 1:
            y[index] = np.array([254,0,0]) / 255.  
        if item == 2:
            y[index] = np.array([255,255,255]) / 255.  
        if item == 3:
            y[index] = np.array([176,48,96]) / 255. 
        if item == 4:
            y[index] = np.array([255,255,0]) / 255.  
        if item == 5:
            y[index] = np.array([255,127,80]) / 255.
        if item == 6:
            y[index] = np.array([0,255,1]) / 255.  
        if item == 7:
            y[index] = np.array([0,205,0]) / 255.  
        if item == 8:
            y[index] = np.array([2,127,1]) / 255.  
        if item == 9:
            y[index] = np.array([126,255,212]) / 255.  
        if item == 10:
            y[index] = np.array([160,32,239]) / 255.  
        if item == 11:
            y[index] = np.array([215,191,215]) / 255.  
        if item == 12:
            y[index] = np.array([0,0,254]) / 255.  
        if item == 13:
            y[index] = np.array([1,0,138]) / 255. 
        if item == 14:
            y[index] = np.array([218,112,213]) / 255.  
        if item == 15:
            y[index] = np.array([160,82,44]) / 255.  
        if item == 16:
            y[index] = np.array([0,255,255]) / 255. 
        if item == 17:
            y[index] = np.array([254,165,0]) / 255.  
        if item == 18:
            y[index] = np.array([128,255,0]) / 255. 
        if item == 19:
            y[index] = np.array([138,139,1]) / 255. 
        if item == 20:
            y[index] = np.array([0,138,138]) / 255.  
        if item == 21:
            y[index] = np.array([205,181,205]) / 255.  
        if item == 22:
            y[index] = np.array([239,154,1]) / 255.  
        if item == -1:
            y[index] = np.array([0, 0, 0]) / 255. 
    return y


def list_to_colormap4(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.  
        if item == 1:
            y[index] = np.array([252,1,1]) / 255. 
        if item == 2:
            y[index] = np.array([0,0,254]) / 255.  
        if item == 3:
            y[index] = np.array([1,255,0]) / 255.  
        if item == 4:
            y[index] = np.array([255,0,254]) / 255.  
        if item == 5:
            y[index] = np.array([255,255,255]) / 255.  
        if item == 6:
            y[index] = np.array([255,254,1]) / 255.  
        if item == 7:
            y[index] = np.array([2,254,255]) / 255.  
        if item == 8:
            y[index] = np.array([47,138,86]) / 255.  
        if item == -1:
            y[index] = np.array([0, 0, 0]) / 255.  
    return y

def classification_map(map, height, width, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(width * 2.0 / dpi, height * 2.0 / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)
    return 0

def generate_png(gt_hsi, pred_hsi, height, width, path):
    gt = gt_hsi.flatten()
    pred = pred_hsi.flatten()
    
    max_label = max(gt.max(), pred.max())
    if max_label == 16:
        y_gt = list_to_colormap1(gt)
        y_pred = list_to_colormap1(pred)

    elif max_label == 8:
        y_gt = list_to_colormap4(gt)
        y_pred = list_to_colormap4(pred)
    elif max_label == 18:
        y_gt = list_to_colormap3(gt)
        y_pred = list_to_colormap3(pred)
    else:
        y_gt = list_to_colormap2(gt)
        y_pred = list_to_colormap2(pred)
    gt_re = np.reshape(y_gt, (height, width, 3))
    pred_re = np.reshape(y_pred, (height, width, 3))
    classification_map(pred_re, height, width, 300, f"{path}_pred.png")
    classification_map(gt_re, height, width, 300, f"{path}_true.png")
    
    print('------Get classification maps successful-------')


