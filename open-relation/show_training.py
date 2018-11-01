import pickle
import matplotlib.pyplot as plt
from train_config import hyper_params

config = hyper_params['pascal']
with open(config['log_loss_path'], 'rb') as log_loss:
    loss = pickle.load(log_loss)
with open(config['log_acc_path'], 'rb') as log_acc:
    acc = pickle.load(log_acc)

loss_x = [i for i in range(0, len(loss))]
plt.plot(loss_x, loss)
plt.title('Training Loss')
plt.xlabel('time')
plt.ylabel('Loss')
plt.show()

acc_x = [i for i in range(0, len(acc))]
plt.plot(acc_x, acc)
plt.title('Training Acc')
plt.xlabel('time')
plt.ylabel('Acc')
plt.show()