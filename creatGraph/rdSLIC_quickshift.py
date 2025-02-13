import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from skimage.segmentation import mark_boundaries, quickshift
from sklearn import preprocessing
from skimage.color import rgb2gray
from loadData import data_reader
def load_data():
    data = data_reader.WHU_Hi_HanChuan().normal_cube
    data_gt = data_reader.WHU_Hi_HanChuan().truth

    return data, data_gt
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
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.  # 背景色
        if item == 1:
            y[index] = np.array([254,0,0]) / 255.  # 红色
        if item == 2:
            y[index] = np.array([255,255,255]) / 255.  # 橙色
        if item == 3:
            y[index] = np.array([176,48,96]) / 255.  # 黄色
        if item == 4:
            y[index] = np.array([255,255,0]) / 255.  # 黄绿色
        if item == 5:
            y[index] = np.array([255,127,80]) / 255.  # 绿色
        if item == 6:
            y[index] = np.array([0,255,1]) / 255.  # 青绿色
        if item == 7:
            y[index] = np.array([0,205,0]) / 255.  # 青色
        if item == 8:
            y[index] = np.array([2,127,1]) / 255.  # 蓝色
        if item == 9:
            y[index] = np.array([126,255,212]) / 255.  # 深蓝色
        if item == 10:
            y[index] = np.array([160,32,239]) / 255.  # 紫罗兰色
        if item == 11:
            y[index] = np.array([215,191,215]) / 255.  # 紫色
        if item == 12:
            y[index] = np.array([0,0,254]) / 255.  # 粉红色
        if item == 13:
            y[index] = np.array([1,0,138]) / 255.  # 浅红色
        if item == 14:
            y[index] = np.array([218,112,213]) / 255.  # 深橙色
        if item == 15:
            y[index] = np.array([160,82,44]) / 255.  # 桃红色
        if item == 16:
            y[index] = np.array([0,255,255]) / 255.  # 淡紫色
        if item == 17:
            y[index] = np.array([254,165,0]) / 255.  # 黄色
        if item == 18:
            y[index] = np.array([128,255,0]) / 255.  # 深紫色
        if item == 19:
            y[index] = np.array([138,139,1]) / 255.  # 深蓝色
        if item == 20:
            y[index] = np.array([0,138,138]) / 255.  # 深紫罗兰色
        if item == 21:
            y[index] = np.array([205,181,205]) / 255.  # 棕色
        if item == 22:
            y[index] = np.array([239,154,1]) / 255.  # 靛蓝色
        if item == -1:
            y[index] = np.array([0, 0, 0]) / 255.  # 处理无效标签
    return y
def SegmentsLabelProcess(labels):
    labels = np.array(labels, np.int64)
    H, W = labels.shape
    ls = list(set(np.reshape(labels, [-1]).tolist()))

    dic = {}
    for i in range(len(ls)):
        dic[ls[i]] = i

    new_labels = labels
    for i in range(H):
        for j in range(W):
            new_labels[i, j] = dic[new_labels[i, j]]
    return new_labels

class QuickshiftSegmentation(object):
    def __init__(self, HSI, labels, kernel_size=3, max_dist=6, ratio=0.5):
        self.kernel_size = kernel_size
        self.max_dist = max_dist
        self.ratio = ratio
        height, width, bands = HSI.shape
        data = np.reshape(HSI, [height * width, bands])
        minMax = preprocessing.StandardScaler()
        data = minMax.fit_transform(data)
        self.data = np.reshape(data, [height, width, bands])
        self.labels = labels

    def get_Q_and_S_and_Segments(self):
        img = self.data
        (h, w, d) = img.shape
        segments = quickshift(img, kernel_size=self.kernel_size, max_dist=self.max_dist, ratio=self.ratio)

        if segments.max() + 1 != len(list(set(segments.ravel()))):
            segments = SegmentsLabelProcess(segments)
        self.segments = segments
        superpixel_count = segments.max() + 1
        self.superpixel_count = superpixel_count

        # out = mark_boundaries(img[:, :, [0, 1, 2]], segments)
        # plt.figure()
        # plt.imshow(out)
        # plt.show()
        #
        # _, data_gt = load_data()
        # width, height = data_gt.shape
        # gt = data_gt.flatten()
        # max_index = np.max(gt)
        # if max_index == 16:
        #     y_gt = list_to_colormap1(gt)
        # else:
        #     y_gt = list_to_colormap2(gt)
        #
        # # pdb.set_trace()
        # gt_re = np.reshape(y_gt, (h, w, 3))
        # # out = mark_boundaries(gt_re, segments,color=(0.68,0.85,0.90))  # 浅蓝色
        #
        # # plt.figure()
        # # plt.imshow(out)
        # # plt.show()
        # fig = plt.figure(frameon=False)
        # fig.set_size_inches(width * 2.0 / 300, height * 2.0 / 300)
        # ax = plt.Axes(fig, [0., 0., 1., 1.])
        # ax.set_axis_off()
        # ax.xaxis.set_visible(False)
        # ax.yaxis.set_visible(False)
        # fig.add_axes(ax)
        # out = mark_boundaries(gt_re, segments)
        # # out = mark_boundaries(img[:, :, [0, 1, 2]], segments)
        # ax.imshow(out)
        #
        # #
        # # out = mark_boundaries(img[:, :, [0, 1, 2]], segments)
        # # plt.figure()
        # # plt.imshow(out)
        # plt.show()
        # fig.savefig('/media/rslab/Workplace/Huanglh/论文_New/HC/quick', dpi=300)

        segments = np.reshape(segments, [-1])
        S = np.zeros([superpixel_count, d], dtype=np.float32)
        Q = np.zeros([w * h, superpixel_count], dtype=np.float32)
        x = np.reshape(img, [-1, d])
        for i in range(superpixel_count):
            idx = np.where(segments == i)[0]
            count = len(idx)
            pixels = x[idx]
            superpixel = np.sum(pixels, 0) / count
            S[i] = superpixel
            Q[idx, i] = 1

        self.S = S
        self.Q = Q
        return Q, S, self.segments

    def get_A(self, sigma: float):
        A = np.zeros([self.superpixel_count, self.superpixel_count], dtype=np.float32)
        (h, w) = self.segments.shape
        for i in range(h - 2):
            for j in range(w - 2):
                sub = self.segments[i:i + 2, j:j + 2]
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                if sub_max != sub_min:
                    idx1 = sub_max
                    idx2 = sub_min
                    if A[idx1, idx2] != 0:
                        continue

                    pix1 = self.S[idx1]
                    pix2 = self.S[idx2]
                    diss = np.exp(-np.sum(np.square(pix1 - pix2)) / sigma ** 2)
                    A[idx1, idx2] = A[idx2, idx1] = diss

        return A

    def get_A_cosine(self):
        A = np.zeros([self.superpixel_count, self.superpixel_count], dtype=np.float32)
        (h, w) = self.segments.shape
        for i in range(h - 2):
            for j in range(w - 2):
                sub = self.segments[i:i + 2, j:j + 2]
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                if sub_max != sub_min:
                    idx1 = sub_max
                    idx2 = sub_min
                    if A[idx1, idx2] != 0:
                        continue

                    pix1 = self.S[idx1]
                    pix2 = self.S[idx2]
                    cosine_sim = np.dot(pix1, pix2) / (np.linalg.norm(pix1) * np.linalg.norm(pix2))
                    A[idx1, idx2] = A[idx2, idx1] = cosine_sim

        return A

    def get_A_binary(self):
        A = np.zeros([self.superpixel_count, self.superpixel_count], dtype=np.float32)
        (h, w) = self.segments.shape
        for i in range(h - 2):
            for j in range(w - 2):
                sub = self.segments[i:i + 2, j:j + 2]
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                if sub_max != sub_min:
                    idx1 = sub_max
                    idx2 = sub_min
                    if A[idx1, idx2] != 0:
                        continue

                    pix1 = self.S[idx1]
                    pix2 = self.S[idx2]
                    A[idx1, idx2] = A[idx2, idx1] = 1

        return A

class LDA_SLIC(object):
    def __init__(self, data, labels):
        self.data = data
        self.init_labels = labels
        self.curr_data = data
        self.height, self.width, self.bands = data.shape
        self.x_flatt = np.reshape(data, [self.width * self.height, self.bands])
        self.y_flatt = np.reshape(labels, [self.height * self.width])
        self.labels = labels

    def LDA_Process(self, curr_labels):  # 进行降维
        curr_labels = np.reshape(curr_labels, [-1])
        idx = np.where(curr_labels != 0)[0]
        x = self.x_flatt[idx]
        y = curr_labels[idx]
        lda = LinearDiscriminantAnalysis(n_components=3)
        lda.fit(x, y - 1)
        X_new = lda.transform(self.x_flatt)
        return np.reshape(X_new, [self.height, self.width, -1])

    def SLIC_Process(self, img):
        mysegment = QuickshiftSegmentation(img, self.labels, kernel_size=3, max_dist=5, ratio=0.01)
        # 推荐0.01
        Q, S, Segments = mysegment.get_Q_and_S_and_Segments()
        A_euclidean = mysegment.get_A(sigma=10)
        A_cosine = mysegment.get_A_cosine()
        A_binary = mysegment.get_A_binary()
        return Q, S, A_euclidean, A_cosine, A_binary, Segments

    def simple_superpixel(self, scale):
        curr_labels = self.init_labels
        X = self.LDA_Process(curr_labels)
        Q, S, A_euclidean, A_cosine, A_binary, Seg = self.SLIC_Process(X)
        return Q, S, A_euclidean, A_cosine, A_binary, Seg

