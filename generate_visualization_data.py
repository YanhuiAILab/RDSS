import numpy as np
import pickle


np.random.seed(42)

# number of data points
n_points = 5000

# number of distributions
n_distributions = 10

data_points = []
labels = []


# For each distribution, we randomly select the mean and standard deviation and sample 500 data points from a normal distribution
for i in range(n_distributions):
    mean_x = np.random.uniform(-12, 12)  
    mean_y = np.random.uniform(-12, 12)  
    std_dev_x = np.random.uniform(0.5, 2)  
    std_dev_y = np.random.uniform(0.5, 2)
    x_points = np.random.normal(mean_x, std_dev_x, n_points // n_distributions)
    y_points = np.random.normal(mean_y, std_dev_y, n_points // n_distributions)
    data_points.extend(list(zip(x_points, y_points)))
    labels.extend([i] * (n_points // n_distributions))

data_points = np.array(data_points)
labels = np.array(labels)

with open("visualization.pkl", "wb") as f:
    pickle.dump((data_points, labels), f)