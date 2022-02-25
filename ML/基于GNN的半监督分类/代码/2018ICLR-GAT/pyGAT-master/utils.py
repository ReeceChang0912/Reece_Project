import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="/mnt/Code/data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    correct_prediction = torch.eq(torch.argmax(output, 1), labels)
    accuracy_all = correct_prediction.float()
    return torch.mean(accuracy_all)

    # preds = output.max(1)[1].type_as(labels)
    # correct = preds.eq(labels).double()
    # correct = correct.sum()
    # return correct / len(labels)

# 图像数据归一化、标准化
def img_normalize(img):
    img = img.astype(np.float32) / 255.0    #归一化为[0.0,1.0]
    means, stdevs = [], []      # 均值，方差
    for i in range(3):
        pixels = img[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))
    means = np.array(means)
    stdevs = np.array(stdevs)

    img = (img - means) / stdevs

    return img


# 图像数据归一化、标准化
def img_normalize_mnist(img):
    img = img.astype(np.float32) / 255.0   #归一化为[0.0,1.0]
    means=np.mean(img)
    stdevs=np.std(img)
    img = (img - means) / stdevs
    # img = (img - 0.1307) / 0.3081
    return img






def build_model_input(train_features, train_ind_label, test_features, test_ind_label, n_class):
    NO_LABEL = -1

    def ind_label2onehot_label(ind_label):
        n_examples = ind_label.shape[0]
        label = np.zeros([n_examples, n_class], dtype=np.uint8)  # label -> (N,c)
        for ind, key in zip(range(n_examples), ind_label):
            if key != NO_LABEL:
                label[ind][key] = 1

        label = torch.from_numpy(label).cuda()
        return label

    def get_label_mask(ind_label):
        label_mask = torch.where(ind_label == NO_LABEL, torch.zeros_like(ind_label), torch.ones_like(ind_label))
        unlabel_mask = torch.sub(1, label_mask)
        return label_mask.cuda(), unlabel_mask.cuda()


    train_label = ind_label2onehot_label(train_ind_label)
    test_label = ind_label2onehot_label(test_ind_label)
    train_label_mask,train_unlabel_mask = get_label_mask(train_ind_label)
    test_label_mask, test_unlabel_mask = get_label_mask(test_ind_label)

    train_input = [train_features, train_label, train_label_mask,train_unlabel_mask]
    test_input = [test_features, test_label, test_label_mask, test_unlabel_mask]

    return train_input, test_input


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
    A = np.zeros((n_obj, n_obj))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            A[node_idx, center_idx] = 1.0
    return A


def get_label_mask(ind_label):
    NO_LABEL = -1
    label_mask = torch.where(ind_label == NO_LABEL, torch.zeros_like(ind_label), torch.ones_like(ind_label))
    unlabel_mask = torch.sub(1, label_mask)
    return label_mask.cuda(), unlabel_mask.cuda()

