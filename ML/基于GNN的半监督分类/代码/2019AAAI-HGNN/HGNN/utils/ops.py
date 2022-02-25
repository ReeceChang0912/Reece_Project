import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp



def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat



def graph_construct(X, k_neig=10):
    """
    param:
        X: N_object x feature_number
        k_neig: the number of neighbor expansion
    return:
        A: N_object x N_object
    """

    dis_mat = Eu_dis(X)
    n_obj = dis_mat.shape[0]
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            H[node_idx, center_idx] = 1.0
    return H





LOSSES_TEXT_PATH = 'result/loss/'
if not osp.exists(LOSSES_TEXT_PATH):
    os.mkdir(LOSSES_TEXT_PATH)

# # 绘制loss曲线
def show_loss(losses, epochs):
    # 创建一个画布
    plt.figure(figsize=(16, 7))
    plt.plot(losses, label='Train Loss')
    plt.title("Train Losses in "+str(epochs)+" epochs")
    # 添加图例
    plt.legend()
    # print("Save losses")
    plt.savefig(LOSSES_TEXT_PATH+'TrainLoss7_'+str(epochs)+'_.png')
    # plt.show()


# 将loss写入文件
def save_loss(losses,  epochs):
    losses = np.array(losses)

    # 将losses写入文件保存
    np.save(LOSSES_TEXT_PATH + "TrainLoss7_" + str(epochs) + ".npy", losses)
    np.savetxt(LOSSES_TEXT_PATH + "TrainLoss7_" + str(epochs) + ".txt", losses)


def save_H(H):
    pass
    # H_np=np.array(H)
    # np.savetxt('result/training_result/4H.txt', H_np)
    # np.save('result/training_result/4H.npy', H_np)


def save_G(G):
    pass
    # G_np=np.array(G)
    # np.savetxt('result/training_result/4G.txt',G_np)
    # np.save('result/training_result/4G.npy', G_np)


def save_Other(O, name):
    pass
    # O_np=np.array(O)
    # np.savetxt('result/training_result/4'+name+'.txt',O_np)
    # np.save('result/training_result/4' + name + '.npy', O_np)