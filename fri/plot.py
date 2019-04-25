import matplotlib

matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import matplotlib.cm as cm

# Get a color for each relevance type
color_palette_3 = cm.Set1([0,1,2],alpha=0.8)


def plot_relevance_bars(ax, ranges, ticklabels=None, classes=None, numbering=True,
                        tick_rotation=30):
    """

    Parameters
    ----------
    ax:
        axis which the bars get drawn on
    ranges:
        the 2d array of floating values determining the lower and upper bounds of the bars
    ticklabels: (optional)
        labels for each feature
    classes: (optional)
        relevance class for each feature, determines color
    numbering: bool
        Add feature index when using ticklabels
    tick_rotation:  int
        Amonut of rotation of ticklabels for easier readability.


    """
    N = len(ranges)

    # Ticklabels
    if ticklabels is None:
        ticks = np.arange(N) + 1
    else:
        ticks = list(ticklabels)
        if numbering:
            for i in range(N):
                ticks[i] += " - {}".format(i + 1)

    # Interval sizes
    ind = np.arange(N) + 1
    width = 0.6
    upper_vals = ranges[:, 1]
    lower_vals = ranges[:, 0]
    height = upper_vals - lower_vals
    # Minimal height to make very small intervals visible
    height[height < 0.001] = 0.001

    # Bar colors
    if classes is None:
        new_classes = np.zeros(N).astype(int)
        color = [color_palette_3[c] for c in new_classes]
    else:
        color = [color_palette_3[c] for c in classes]

    # Plot the bars
    bars = ax.bar(ind, height, width, bottom=lower_vals, tick_label=ticks, align="center", edgecolor=["black"] * N,
                  linewidth=1.3, color=color)

    ax.set_xticklabels(ticks)
    if ticklabels is not None:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=tick_rotation, ha="right")
    # ax.tick_params(rotation="auto")
    # Limit the y range to 0,1 or 0,L1
    ax.set_ylim([0, max(ranges[:, 1]) * 1.1])

    ax.set_ylabel('relevance')
    ax.set_xlabel('feature')

    if classes is not None:
        relevance_classes = ["Irrelevant", "Weakly relevant", "Strongly relevant"]
        patches = []
        for i, rc in enumerate(relevance_classes):
            patch = mpatches.Patch(color=color_palette_3[i], label=rc)
            patches.append(patch)

        ax.legend(handles=patches)

    return bars

def plotIntervals(ranges, ticklabels=None, invert=False, classes=None):
    # Figure Parameters
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(111)
    N = len(ranges)

    out = plot_relevance_bars(ax, ranges, ticklabels=ticklabels, classes=classes)
    fig.autofmt_xdate()
    # Invert the xaxis for cases in which the comparison with other tools
    if invert:
        plt.gca().invert_xaxis()
    return fig


def plot_dendrogram_and_intervals(intervals, linkage, figsize=(13, 7), ticklabels=None, classes=None,
                                  **kwargs):
    fig, (ax2, ax) = plt.subplots(2, 1, figsize=figsize)
    # Top dendrogram plot
    d = dendrogram(
        linkage,
        color_threshold=0,
        leaf_rotation=0.,  # rotates the x axis labels
        leaf_font_size=12.,  # font size for the x axis labels
        ax=ax2
    )
    # Get index determined through linkage method and dendrogram
    rearranged_index = d['leaves']
    ranges = intervals[rearranged_index]

    if ticklabels is None:
        ticks = np.array(rearranged_index)
        ticks += 1  # Index starting at 1
    else:
        ticks = list(ticklabels[rearranged_index])
        for i in range(len(intervals)):
            ticks[i] += " - {}".format(rearranged_index[i] + 1)

    plot_relevance_bars(ax, ranges, ticklabels=ticks,
                        classes=classes[rearranged_index] if classes is not None else None, numbering=False, **kwargs)
    fig.subplots_adjust(hspace=0)
    ax.margins(x=0)
    ax2.set_xticks([])
    ax2.margins(x=0)
    plt.tight_layout()

    return fig


def plot_intervals(model, ticklabels=None):
    """Plot the relevance intervals.
    
    Parameters
    ----------
    model : FRI model
        Needs to be fitted before.
    ticklabels : list of str, optional
        Strs for ticklabels on x-axis (features)
    """
    if model.interval_ is not None:
        plotIntervals(model.interval_, ticklabels=ticklabels, classes=model.relevance_classes_)
    else:
        print("Intervals not computed. Try running fit() function first.")


def interactive_scatter_embed(embedding, mode="markers", txt=None):
    # TODO: extend method
    import plotly.graph_objs as go
    from plotly.offline import init_notebook_mode, iplot
    init_notebook_mode(connected=True)
    # Create a trace
    trace = go.Scatter(
        x=embedding[:, 0],
        y=embedding[:, 1],
        mode=mode,
        text=txt if mode is "text" else None
    )

    data = [trace]

    # Plot and embed in ipython notebook!
    iplot(data)
