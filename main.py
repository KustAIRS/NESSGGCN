import argparse
import numpy as np
from sklearn.metrics import classification_report, cohen_kappa_score, accuracy_score, confusion_matrix
import torch
import time
import yaml
import scipy.io as sio
from model import utils,NESSGGCN
from loadData import data_reader, split_data
from createGraph import rdSLIC, create_graph, rdSLIC_Fel,rdSLIC_quickshift,rdSLIC_watershed
def load_data():
    data = data_reader.WHU_Hi_HongHu().normal_cube
    data_gt = data_reader.WHU_Hi_HongHu().truth
    return data, data_gt


def train_and_evaluate(iteration):
    # load data
    data, data_gt = load_data()
    class_num = np.max(data_gt)
    height, width, bands = data.shape
    gt_reshape = np.reshape(data_gt, [-1])
    
    # load config
    config = yaml.load(open(args.path_config, "r"), Loader=yaml.FullLoader)
    dataset_name = config["data_input"]["dataset_name"]
    samples_type = config["data_split"]["samples_type"]
    train_num = config["data_split"]["train_num"]
    train_ratio = config["data_split"]["train_ratio"]
    superpixel_scale = config["data_split"]["superpixel_scale"]
    max_epoch = config["network_config"]["max_epoch"]
    learning_rate = config["network_config"]["learning_rate"]
    weight_decay = config["network_config"]["weight_decay"]
    lb_smooth = config["network_config"]["lb_smooth"]
    path_weight = config["result_output"]["path_weight"]
    path_result = config["result_output"]["path_result"]

    if args.print_config:
        print(config)

    # Dataset split
    train_index, test_index = split_data.split_data(gt_reshape, class_num, train_ratio, train_num, samples_type)
    train_samples_gt, test_samples_gt = create_graph.get_label(gt_reshape, train_index, test_index)
    train_label_mask, test_label_mask = create_graph.get_label_mask(train_samples_gt, test_samples_gt, data_gt, class_num)

    train_gt = np.reshape(train_samples_gt, [height, width])
    test_gt = np.reshape(test_samples_gt, [height, width])

    if args.print_data_info:
        data_reader.data_info(train_gt, test_gt)

    train_gt_onehot = create_graph.label_to_one_hot(train_gt, class_num)
    test_gt_onehot = create_graph.label_to_one_hot(test_gt, class_num)

    # Superpixel segmentation (Either one can be selected)
    ls = rdSLIC.LDA_SLIC(data, train_gt, class_num - 1)
    ls1 = rdSLIC_Fel.LDA_SLIC(data, train_gt)
    ls2 = rdSLIC_quickshift.LDA_SLIC(data, train_gt)
    ls3 = rdSLIC_watershed.LDA_SLIC(data, train_gt)


    tic0 = time.time()

    # Modify according to your choice
	Q, S, A_euclidean, A_cosine, A_binary, Seg = ls.simple_superpixel(scale=superpixel_scale)
    
	toc0 = time.time()
    LDA_SLIC_Time = toc0 - tic0
    print('Q shape:', Q.shape)

    Q = torch.from_numpy(Q).to(args.device)
    A_euclidean = torch.from_numpy(A_euclidean).to(args.device)
    A_cosine = torch.from_numpy(A_cosine).to(args.device)
    A_binary = torch.from_numpy(A_binary).to(args.device)
    train_samples_gt = torch.from_numpy(train_samples_gt.astype(np.float32)).to(args.device)
    test_samples_gt = torch.from_numpy(test_samples_gt.astype(np.float32)).to(args.device)
    train_gt_onehot = torch.from_numpy(train_gt_onehot.astype(np.float32)).to(args.device)
    test_gt_onehot = torch.from_numpy(test_gt_onehot.astype(np.float32)).to(args.device)
    train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(args.device)
    test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(args.device)
    net_input = torch.from_numpy(np.array(data, np.float32)).to(args.device)

    # Model initialization
    net = NESSGGCN.NESSGGCN(height, width, bands, class_num, Q, A_euclidean, A_binary).to(args.device)
    print('net info', net)

    # train
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    zeros = torch.zeros([height * width]).to(args.device).float()
    best_loss = 99999999
    net.train()
    tic1 = time.time()
    for i in range(max_epoch + 1):
        optimizer.zero_grad()
        # pdb.set_trace()
        output = net(net_input)
        loss = utils.compute_loss(output, train_gt_onehot, train_label_mask)
        loss.backward(retain_graph=False)
        optimizer.step()
        with torch.no_grad():
            net.eval()
            output = net(net_input)
            trainloss = utils.compute_loss(output, train_gt_onehot, train_label_mask)
            trainOA = utils.evaluate_performance(output, train_samples_gt, train_gt_onehot, zeros)
            testloss = utils.compute_loss(output, test_gt_onehot, test_label_mask)
            testOA = utils.evaluate_performance(output, test_samples_gt, test_gt_onehot, zeros)
            if testloss < best_loss:
                best_loss = testloss
                torch.save(net.state_dict(), path_weight + r"model.pt")
        torch.cuda.empty_cache()
        net.train()
        if i % 10 == 0:
            print("{}\ttrain loss={:.4f}\t train OA={:.4f} test loss={:.4f}\t test OA={:.4f}".format(str(i + 1), trainloss, trainOA, testloss, testOA))
    toc1 = time.time()

    # test
    torch.cuda.empty_cache()
    with torch.no_grad():
        net.load_state_dict(torch.load(path_weight + r"model.pt"))
        net.eval()
        tic2 = time.time()
        output = net(net_input)
        toc2 = time.time()
        testloss = utils.compute_loss(output, test_gt_onehot, test_label_mask)
        testOA = utils.evaluate_performance(output, test_samples_gt, test_gt_onehot, zeros)
        
        # Plot output results
        pre_labels = torch.argmax(output, dim=1) + 1  
        print('pre_label shape', pre_labels.shape)
        background_indices = (gt_reshape == 0)
        pre_labels[background_indices] = 0  
        pre_labels_np = pre_labels.cpu().numpy()
        print('class_min:', np.min(pre_labels_np))
        print('class_max:', np.max(pre_labels_np))
        pre_labels_np = np.reshape(pre_labels_np, [height, width])
        
        run_date1 = time.strftime('%Y%m%d-%H%M-', time.localtime(time.time()))
        data_reader.generate_png(data_gt, pre_labels_np, height, width, f"your path/pic/{run_date1}pic_{iteration}.png")
        sio.savemat(f'your path/results/{run_date1}output_{iteration}.mat', {'output': output.cpu().numpy()})
        sio.savemat(f'your path/results/{run_date1}pre_labels_np_{iteration}.mat', {'pre_labels_np': pre_labels_np})

    training_time = toc1 - tic1 + LDA_SLIC_Time
    testing_time = toc2 - tic2 + LDA_SLIC_Time
    test_label_mask_cpu = test_label_mask.cpu().numpy()[:, 0].astype('bool')
    test_samples_gt_cpu = test_samples_gt.cpu().numpy().astype('int64')
    predict = torch.argmax(output, 1).cpu().numpy()
    classification = classification_report(test_samples_gt_cpu[test_label_mask_cpu],
                                           predict[test_label_mask_cpu] + 1, digits=4)
    kappa = cohen_kappa_score(test_samples_gt_cpu[test_label_mask_cpu], predict[test_label_mask_cpu] + 1)
    overall_acc = accuracy_score(test_samples_gt_cpu[test_label_mask_cpu], predict[test_label_mask_cpu] + 1)
    confusion_matrix1 = confusion_matrix(test_samples_gt_cpu[test_label_mask_cpu], predict[test_label_mask_cpu] + 1)
    each_acc, average_acc = utils.aa_and_each_accuracy(confusion_matrix1)


    return overall_acc, average_acc, kappa, each_acc, training_time, testing_time

def record_output(oa_ae, aa_ae, kappa_ae, element_acc_ae, training_time_ae, testing_time_ae, path):
    f = open(path, 'a')
    sentence0 = 'OAs for each iteration are:' + str(oa_ae) + '\n'
    f.write(sentence0)
    sentence1 = 'AAs for each iteration are:' + str(aa_ae) + '\n'
    f.write(sentence1)
    sentence2 = 'KAPPAs for each iteration are:' + str(kappa_ae) + '\n' + '\n'
    f.write(sentence2)
    sentence3 = 'mean_OA ± std_OA is: ' + str(np.mean(oa_ae)) + '±' + str(np.std(oa_ae)) + '\n'
    f.write(sentence3)
    sentence4 = 'mean_AA ± std_AA is: ' + str(np.mean(aa_ae)) + '±' + str(np.std(aa_ae)) + '\n'
    f.write(sentence4)
    sentence5 = 'mean_KAPPA ± std_KAPPA is: ' + str(np.mean(kappa_ae)) + '±' + str(np.std(kappa_ae)) + '\n' + '\n'
    f.write(sentence5)
    sentence6 = 'Total average Training time is: ' + str(np.mean(training_time_ae)) + '\n'
    f.write(sentence6)
    sentence7 = 'Total average Testing time is: ' + str(np.mean(testing_time_ae)) + '\n' + '\n'
    f.write(sentence7)
    element_mean = np.mean(element_acc_ae, axis=0)
    element_std = np.std(element_acc_ae, axis=0)
    sentence8 = "Mean of all elements in confusion matrix: " + str(element_mean) + '\n'
    f.write(sentence8)
    sentence9 = "Standard deviation of all elements in confusion matrix: " + str(element_std) + '\n'
    f.write(sentence9)
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FDGC')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--path-config', type=str, default='your path/config/config.yaml')
    parser.add_argument('-pc', '--print-config', action='store_true', default=False)
    parser.add_argument('-pdi', '--print-data-info', action='store_true', default=True)
    parser.add_argument('-sr', '--show-results', action='store_true', default=True)
    parser.add_argument('--save-results', action='store_true', default=True)
    args = parser.parse_args()

    iterations = 5
    oa_ae, aa_ae, kappa_ae, element_acc_ae, training_time_ae, testing_time_ae = [], [], [], [], [], []

    for i in range(iterations):
        oa, aa, kappa, element_acc, train_time, test_time = train_and_evaluate(i)
        # pdb.set_trace()
        oa_ae.append(oa)
        aa_ae.append(aa)
        kappa_ae.append(kappa)
        element_acc_ae.append(element_acc)
        training_time_ae.append(train_time)
        testing_time_ae.append(test_time)

    utils.record_output(oa_ae, aa_ae, kappa_ae, element_acc_ae, training_time_ae, testing_time_ae, 'your path/summary.txt')
