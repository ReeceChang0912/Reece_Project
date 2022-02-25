import numpy as np


Array_acc_val = [93.47,93.31,94.25,93.78,94.54,94.11,95.01,93.76,94.91,94.86
]

mean_acc = np.mean(Array_acc_val)
std_acc = np.std(Array_acc_val)
ci95 = 1.96 * std_acc / np.sqrt(len(Array_acc_val))
print('acc:{:.4f}, std:{:.4f}, ci95:{:.4f}'.format(mean_acc, std_acc, ci95))

