import numpy as np
import matplotlib.pyplot as plt


def plot_contours(f, x_limits, y_limits, title="Loss Function Contour", path=None, path_labels=None, filename=None):
    # Create grid / plot space based on limits input
    x = np.linspace(x_limits[0], x_limits[1], 500)
    y = np.linspace(y_limits[0], y_limits[1], 500)
    X, Y = np.meshgrid(x, y)

    # Get function values for points on plot space / grid
    Z = np.array([[f(np.array([X[i, j], Y[i, j]]))[0] for j in range(X.shape[1])] for i in range(X.shape[0])])

    plt.figure(figsize=(16, 12))

    # Handle linear differently without log scale/spacing for contours
    if "Linear" in title:
        levels = np.linspace(np.min(Z), np.max(Z), 30)
    else: # not linear, log scale/spacing for visualization
        levels = np.logspace(np.log10(np.min(Z)), np.log10(np.max(Z)), 30)

    # plot contour lines
    cp = plt.contour(X, Y, Z, levels=levels, cmap='cividis')

    plt.colorbar(cp)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')

    # Plot optimization paths per function
    if path is not None and path_labels is not None:
        for p, label in zip(path, path_labels):
            if "Newton" in label:
                plt.plot(p[:, 0], p[:, 1], marker='o', label=label, alpha=0.8)
            else:
                plt.plot(p[:, 0], p[:, 1], marker='x', label=label, markersize = 10)
        plt.legend()

    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_function_values(iteration_data, labels, filename=None):
    plt.figure(figsize=(16, 12))

    # Plot function values
    for data, label in zip(iteration_data, labels):
        if "Newton" in label:
            plt.plot(data, label=label, marker='o', alpha = 0.8)
        else:
            plt.plot(data, label=label, marker='x', markersize = 10)
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.title('Function Values by Iteration')
    plt.legend()
    plt.grid(True)

    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

