""" need to finish this part """
import matplotlib.pyplot as plt
from matplotlib import colors


def col_histogram(array, n_bins):
    """function that creates a simple colored histogram"""
    N, bins, patches = plt.hist(array, bins=n_bins)
    fracs = N / N.max()
    # we need to normalize the data to 0..1 for the full range of the colormap
    norm = colors.Normalize(fracs.min(), fracs.max())
    # Now, we'll loop through our objects and set the color of each accordingly
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)
    plt.show()
