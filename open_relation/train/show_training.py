import pickle
import matplotlib.pyplot as plt
from train_config import hyper_params

dataset = 'vrd'
target = 'predicate'

config = hyper_params[dataset][target]
with open(config['log_loss_path'], 'rb') as log_loss:
    loss = pickle.load(log_loss)
with open(config['log_acc_path'], 'rb') as log_acc:
    acc = pickle.load(log_acc)

# start = max(0, len(acc) - 2000)
start = 0

loss_x = [i for i in range(start, len(loss))]
plt.plot(loss_x, loss[start:])
plt.title('Training Loss')
plt.xlabel('time')
plt.ylabel('Loss')
plt.show()

acc_x = [i for i in range(start, len(acc))]
plt.plot(acc_x, acc[start:])
plt.title('Training Acc')
plt.xlabel('time')
plt.ylabel('Acc')
plt.show()