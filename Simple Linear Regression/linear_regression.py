import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15
SHUFFLE = True
LEARNING_RATE = 0.01
NUM_ITERS = 200
INPUT_DIM = 1
OUTPUT_DIM = 1


def get_data(num_samples):
    # generates an array of values between 0-49
    X = np.array(range(num_samples))
    # generates size num_samples random numbers from -10-20 
    random_noise = np.random.uniform(-10, 20, size = num_samples)
    y = 3.5 * X + random_noise
    # set y to have 2 decimal points and without scientific notation
    np.set_printoptions(precision = 2, suppress = True)
    return X, y


def standardize(data, mean, std):
    '''
    z = xi - mean / standard deviation
    transforms variables to the same scale so they all contribute equally to the analysis
    
    '''
    return (data - mean) / std


# compare predictions with the actual y value
def cost(N, y, y_pred):
    return (1 / N) * np.sum((y - y_pred)**2)


# gradient descent to adjust weights
def derivates(N, X, y, y_pred):
    dW = -(2/N) * np.sum((y - y_pred) * X)
    db = -(2/N) * np.sum((y - y_pred))
    return dW, db


# update the weight and bias
def update_function(W, b, learning_rate, dW, db):
    W +=  -learning_rate * dW
    b += -learning_rate * db
    return W, b


data = pd.read_csv('Salary_Data.csv')
X = data.iloc[:, 0].values
X = np.reshape(X, (-1, 1))
y = data.iloc[:, 1].values
y = np.reshape(y, (-1, 1))


NUM_SAMPLES = len(X)


# shuffle the data
if SHUFFLE:
    indices = list(range(NUM_SAMPLES))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

  
train_start = 0
train_end = int(0.7 * NUM_SAMPLES)
val_start = train_end
val_end = int((TRAIN_SIZE + VAL_SIZE) * NUM_SAMPLES)
test_start = val_end


# split the data
X_train = X[train_start:train_end]
y_train = y[train_start:train_end]
X_val = X[val_start:val_end]
y_val = y[val_start:val_end]
X_test = X[test_start:]
y_test = y[test_start:]


# get means and stds
X_mean = np.mean(X_train)
X_std = np.std(X_train)
y_mean = np.mean(y_train)
y_std = np.std(y_train)

# standardize
X_train = standardize(X_train, X_mean, X_std)
y_train = standardize(y_train, y_mean, y_std)
X_val = standardize(X_val, X_mean, X_std)
y_val = standardize(y_val, y_mean, y_std)
X_test = standardize(X_test, X_mean, X_std)
y_test = standardize(y_test, y_mean, y_std)


N_train = len(X_train)
N_test = len(X_test)

# initialize random weights
W = 0.01 * np.random.randn(INPUT_DIM, OUTPUT_DIM)
b = np.zeros((1, 1))

for iter in range(NUM_ITERS):
    # get prediction
    y_pred = np.dot(X_train, W) + b
    #calculate loss
    loss = cost(N_train, y_train, y_pred)
    # show progress
    if iter % 10 == 0:
        print(f"Iteration: {iter}, loss: {loss:.3f}")
    # get derviates
    dW, db = derivates(N_train, X_train, y_train, y_pred)
    # update parameters
    W, b = update_function(W, b, LEARNING_RATE, dW, db)

# predictions
pred_train = W * X_train + b
pred_test = W * X_test + b

# cost for train and test
train_cost = cost(N_train, y_train, pred_train)
test_cost = cost(N_test, y_test, pred_test)

print(f"train_cost = {train_cost}, test_cost = {test_cost}")


plt.figure(figsize=(15, 5))

# plot train data
plt.subplot(1, 2, 1)
plt.title('Train')
plt.scatter(X_train, y_train, label = 'y_train')
plt.plot(X_train, pred_train, color = 'red', linewidth = 1, linestyle = '-', label = 'model')
plt.legend(loc='lower right')

#plot test data
plt.subplot(1, 2, 2)
plt.title('Test')
plt.scatter(X_test, y_test, label = 'y_test')
plt.plot(X_test, pred_test, color = 'red', linewidth = 1, linestyle = '-', label = 'model')
plt.legend(loc = 'lower right')

# show plots
plt.show()