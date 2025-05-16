import matplotlib.pyplot as plt
import numpy as np
import prettytable as pt
from sklearn.metrics import confusion_matrix


# Taken from stackoverflow: https://stackoverflow.com/questions/5283649/plot-smooth-line-with-pyplot
def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def smooth_plot(data, length=0, weight=0.9, title=""):
    plt.figure(figsize=(10, 5))
    plt.plot(data[-length:], 'r')
    plt.plot(np.array(smooth(data, weight))[-length:], 'b')
    plt.title(title)
    plt.show()


def confusion_matrix_pretty(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    x = pt.PrettyTable()
    x.field_names = [""] + labels
    for i, row in enumerate(cm):
        x.add_row([labels[i]] + list(row))
    return x.get_string()
