import numpy as np


Array_acc_val = [88.82,88.66,88.21,88.24,89.72,88.71,89.18,88.68,88.69,88.09
]

mean_acc = np.mean(Array_acc_val)
std_acc = np.std(Array_acc_val)
ci95 = 1.96 * std_acc / np.sqrt(len(Array_acc_val))
print('acc:{:.4f}, std:{:.4f}, ci95:{:.4f}'.format(mean_acc, std_acc, ci95))

