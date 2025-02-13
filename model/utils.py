import torch
import numpy as np
from operator import truediv

def compute_loss(network_output: torch.Tensor, train_samples_gt_onehot: torch.Tensor, train_label_mask: torch.Tensor):
    real_labels = train_samples_gt_onehot
    we = -torch.mul(real_labels,torch.log(network_output))
    we = torch.mul(we, train_label_mask)
    pool_cross_entropy = torch.sum(we)
    return pool_cross_entropy

def evaluate_performance(network_output, train_samples_gt, train_samples_gt_onehot, zeros):
    with torch.no_grad():
        available_label_idx = (train_samples_gt!=0).float()        # 有效标签的坐标,用于排除背景
        available_label_count = available_label_idx.sum()          # 有效标签的个数
        correct_prediction = torch.where(torch.argmax(network_output, 1) ==torch.argmax(train_samples_gt_onehot, 1), available_label_idx, zeros).sum()
        OA= correct_prediction.cpu() / available_label_count
        return OA

def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc



def record_output(oa_ae, aa_ae, kappa_ae, element_acc_ae, training_time_ae, testing_time_ae, path):
    with open(path, 'a') as f:
        f.write('OAs for each iteration are:' + str(np.round(oa_ae, 4)*100) + '\n')
        f.write('AAs for each iteration are:' + str(np.round(aa_ae, 4)*100) + '\n')
        f.write('KAPPAs for each iteration are:' + str(np.round(kappa_ae, 4)*100) + '\n\n')
        f.write('mean_OA±std_OA is: {:.2f}±{:.2f}\n'.format(np.mean(oa_ae)*100, np.std(oa_ae)*100))
        f.write('mean_AA±std_AA is: {:.2f}±{:.2f}\n'.format(np.mean(aa_ae) * 100, np.std(aa_ae) * 100))
        f.write('mean_KAPPA±std_KAPPA is: {:.2f}±{:.2f}\n'.format(np.mean(kappa_ae)*100, np.std(kappa_ae)*100))
        f.write('Total average Training time is: ' + str(np.mean(training_time_ae)) + '\n')
        f.write('Total average Testing time is: ' + str(np.mean(testing_time_ae)) + '\n\n')
        element_mean = np.mean(element_acc_ae, axis=0)*100
        element_std = np.std(element_acc_ae, axis=0)*100
        f.write("Mean of all elements in confusion matrix: " + str(np.round(element_mean, 2)) + '\n')
        f.write("Standard deviation of all elements in confusion matrix: " + str(np.round(element_std, 2)) + '\n')
        for i in range(len(element_mean)):
            sentence = "Class {}: {:.2f}±{:.2f}\n".format(i+1, element_mean[i], element_std[i])
            f.write(sentence)
