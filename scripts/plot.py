import matplotlib.pyplot as plt


# Epoch   1/30 | Loss: 0.8101 | Acc: 57.68% | Time: 155610ms | Val Loss: 1.4818 | Val Acc: 50.00%
# Epoch   2/30 | Loss: 0.9108 | Acc: 49.11% | Time: 160731ms | Val Loss: 1.3614 | Val Acc: 60.00%
# Epoch   3/30 | Loss: 0.8846 | Acc: 48.93% | Time: 153107ms | Val Loss: 1.4613 | Val Acc: 50.00%
# Epoch   4/30 | Loss: 0.8562 | Acc: 50.71% | Time: 163154ms | Val Loss: 1.5970 | Val Acc: 50.00%
# Epoch   5/30 | Loss: 0.8678 | Acc: 59.64% | Time: 151629ms | Val Loss: 1.2508 | Val Acc: 50.00%
# Epoch   6/30 | Loss: 0.8436 | Acc: 57.32% | Time: 151370ms | Val Loss: 1.3394 | Val Acc: 50.00%
# Epoch   7/30 | Loss: 0.9249 | Acc: 58.57% | Time: 149938ms | Val Loss: 1.2351 | Val Acc: 50.00%
# Epoch   8/30 | Loss: 0.8656 | Acc: 57.68% | Time: 142314ms | Val Loss: 1.5512 | Val Acc: 50.00%

x = range(8)
y = [57.68, 49.11, 48.93, 50.71, 59.64, 57.32, 58.57, 57.68]
plt.plot(x, y)
plt.show()