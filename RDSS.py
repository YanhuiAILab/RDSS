import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
import pickle
from tqdm import tqdm


def gaussiankernel(data1, data2=None, gamma=0.5):
    if data2 is None:
        d = pairwise_distances(data1, squared=True)
        return np.exp(-gamma * d)
    else:
        d = cdist(data1, data2, 'sqeuclidean')
        return np.exp(-gamma * d)


def khsample(data, kernel, n, alpha):
    N = data.shape[0]
    S = []
    all_indices = list(range(N))
    mu = kernel(data)
    mu = mu - np.diag(np.diag(mu))
    mu = np.sum(mu, axis=0) / N
    F = np.zeros(N)
    for i in tqdm(range(n)):
        idx = np.argmin(F - alpha * mu)
        # Remove index from all_indices and add to S
        S.append(all_indices.pop(idx))
        temp_data = data[idx:idx+1, :]
        data = np.delete(data, idx, axis=0)
        mu = np.delete(mu, idx)
        F = np.delete(F, idx)
        for j in range(N - i - 1):
            F[j] = (1 - 1 / (i+1)) * F[j] + 1 / (i+1) * \
                kernel(np.asarray(temp_data), data[j:j+1, :])
    return S, all_indices  # Return sampled and not sampled indices


if __name__ == '__main__':
    # Number of samples to select in cifar10
    num_samples_cifar10 = [40, 250, 4000]

    # Number of samples to select in cifar100
    num_samples_cifar100 = [400, 2500, 10000]

    # Number of samples to select in svhn
    num_samples_svhn = [40, 250, 1000]

    # sample selection in CIFAR-10
    with open('./image_embeddings_cifar_10.pkl', 'rb') as f:
        data_dict1 = pickle.load(f)
    data1 = np.array([v for v in data_dict1.values()])

    for num_samples in num_samples_cifar10:
        alpha = 1 - 1 / (num_samples**0.5)
        sampled_indices, not_sampled_indices = khsample(
            data1, gaussiankernel, n=num_samples, alpha=alpha)
        # save sampled indices
        with open('cifar10_labeled_' + str(num_samples) + '_' + str(alpha) + '.txt', 'w') as file:
            for item in sampled_indices:
                file.write("%s\n" % item)

        # save not sampled indices
        with open('cifar10_unlabeled_' + str(num_samples) + '_' + str(alpha) + '.txt', 'w') as file:
            for item in not_sampled_indices:
                file.write("%s\n" % item)
    print('cifar10 done!')

    # sanple selection in CIFAR-100
    with open('./image_embeddings_cifar_100.pkl', 'rb') as f:
        data_dict2 = pickle.load(f)
    data2 = np.array([v for v in data_dict2.values()])

    for num_samples in num_samples_cifar100:
        alpha = 1 - 1 / (num_samples**0.5)
        sampled_indices, not_sampled_indices = khsample(
            data2, gaussiankernel, n=num_samples, alpha=alpha)
        # save sampled indices
        with open('cifar100_labeled_' + str(num_samples) + '_' + str(alpha) + '.txt', 'w') as file:
            for item in sampled_indices:
                file.write("%s\n" % item)

        # save not sampled indices
        with open('cifar100_unlabeled_' + str(num_samples) + '_' + str(alpha) + '.txt', 'w') as file:
            for item in not_sampled_indices:
                file.write("%s\n" % item)
    print('cifar100 done!')

    # sample selection in SVHN
    with open('image_embeddings_svhn_train.pkl', 'rb') as f:
        data_dict3 = pickle.load(f)
    data3 = np.array(data_dict3)

    for num_samples in num_samples_cifar100:
        alpha = 1 - 1 / (num_samples**0.5)
        sampled_indices, not_sampled_indices = khsample(
            data3, gaussiankernel, n=num_samples, alpha=alpha)
        # save sampled indices
        with open('svhn_labeled_' + str(num_samples) + '_' + str(alpha) + '.txt', 'w') as file:
            for item in sampled_indices:
                file.write("%s\n" % item)

        # save not sampled indices
        with open('svhn_unlabeled_' + str(num_samples) + '_' + str(alpha) + '.txt', 'w') as file:
            for item in not_sampled_indices:
                file.write("%s\n" % item)
    print('svhn done!')