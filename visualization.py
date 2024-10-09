import matplotlib.pyplot as plt
import pickle
import random
import numpy as np
from RDSS import gaussiankernel, khsample

seed = 42
num_samples = 100
alpha = 1 - 1 / (num_samples**0.5)
np.random.seed(seed)
random.seed(seed)


def visualization(method):
    indices_path = 'vis_' + method + '.txt'
    
    # load sampling indices
    with open(indices_path, "r") as f:
        black_indices = [int(line.strip()) for line in f]
        
    for i in range(10):  # assume there are 10 distributions
        plt.scatter(data_points[labels == i, 0], data_points[labels == i, 1], label=f'Distribution {i + 1}', s=10)
    
    plt.scatter(data_points[black_indices, 0], data_points[black_indices, 1], color='black', s=10, label='Highlighted Points')
    plt.axis('off')
    plt.savefig(method + '.png', dpi=600)


# load data
with open("visualization.pkl", "rb") as f:
    data_points, labels = pickle.load(f)


# ours
sampled_indices, not_sampled_indices = khsample(data_points, gaussiankernel, n=num_samples, alpha=alpha)
    
# save sampling results
with open('vis_ours.txt', 'w') as file:
    for item in sampled_indices:
        file.write("%s\n" % item)

# visualization
visualization('ours')
print('Ours done!')


# random
selected_integers = np.random.choice(np.arange(5000), size=num_samples, replace=False)

# save sampling results
with open("vis_random.txt", "w") as f:
    for num in selected_integers:
        f.write(str(num) + '\n')
visualization('random')
print('Random done!')


# stratified
nums = list(range(5000))

# 5000 / 10 = 500
chunk_size = 500
chunks = [nums[i:i + chunk_size] for i in range(0, len(nums), chunk_size)]

selected_nums = []
for chunk in chunks:
    selected_nums.extend(random.sample(chunk, 10))

# save sampling results
with open('vis_stratified.txt', 'w') as f:
    for num in selected_nums:
        f.write(str(num) + '\n')
visualization('stratified')
print('Stratified done!')