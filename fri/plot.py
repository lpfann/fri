import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font) 
bmap = sns.color_palette("Set2", 5)
sns.set(style='ticks', palette='Set2')
sns.despine()

def plotIntervals(ranges,ticklabels=None,invert=False):
    #fig = plt.figure(figsize=(13, 6))
    #ax = fig.add_subplot(111)
    fig, ax = plt.subplots()
    N = len(ranges)
    if  ticklabels is None:
        ticks = np.arange(N)+1
    else:
        ticks = list(ticklabels)
        for i in range(N):
            ticks[i] += " - {}".format(i+1)

    
    ind = np.arange(N)+1
    width = 0.6
    upper_vals = ranges[:,1]
    lower_vals = ranges[:,0]
    height = upper_vals - lower_vals
    # Minimal height to make very small intervals visible
    height[height < 0.01] = 0.02
    bars = ax.bar(ind, height, width,bottom=lower_vals,tick_label=ticks,align="center" , edgecolor="none",linewidth=1.3)
    
    plt.xticks(ind,ticks, rotation='vertical')
    # loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
    # ax.xaxis.set_major_locator(loc)
    #ax.set_ylim([0,1])
    #ax.set_xlim([0,33])
    plt.ylabel('relevance',fontsize=19)
    plt.xlabel('feature',fontsize=19)
    if invert:
        plt.gca().invert_xaxis()
    return fig

def plot_dendrogram_and_intervals(intervals,linkage,threshold=0.55,ticklabels=None):
    z = linkage
    fig = plt.figure(figsize=(13, 6))

    # Top dendrogram plot
    ax2 = fig.add_subplot(211)
    d = dendrogram(
        z,
        color_threshold=threshold,
        leaf_rotation=0.,  # rotates the x axis labels
        leaf_font_size=12.,  # font size for the x axis labels
        ax=ax2
    )
    # Get index determined through linkage method and dendrogram
    rearranged_index = d['leaves']
    ranges = intervals[rearranged_index]
    
    ax = fig.add_subplot(212)
    N = len(ranges)
    if  ticklabels is None:
        ticks = np.array(rearranged_index)
        ticks +=1 # Index starting at 1
    else:
        ticks = list(ticklabels[rearranged_index])
    
    ind = np.arange(N)+1
    width = 0.6
    upper_vals = ranges[:,1]
    lower_vals = ranges[:,0]
    bars = ax.bar(ind, upper_vals - lower_vals, width,bottom=lower_vals,tick_label=ticks,align="center" ,linewidth=1.3)

    plt.ylabel('relevance',fontsize=19)
    plt.xlabel('feature',fontsize=19)
    plt.xticks(ind,ticks, rotation='vertical')
    ax.margins(x=0)
    ax2.set_xticks([])
    ax2.margins(x=0)
    plt.tight_layout()
    #plt.subplots_adjust(wspace=0, hspace=0)
    
    return fig