import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as plticker
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib


font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font) 
bmap = sns.color_palette("Set2", 5)
sns.set(style='ticks', palette='Set2')
sns.despine()

def plotIntervals(ranges,ticklables=None):
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(111)

    N = len(ranges)
    if  ticklables is None:
        ticks = np.arange(N)+1
    else:
        ticks = list(ticklables)
        for i in range(N):
            ticks[i] += " - {}".format(i+1)

    
    ind = np.arange(N)+1
    width = 0.6
    upper_vals = ranges[:,1]
    lower_vals = ranges[:,0]

    ax.bar(ind, upper_vals - lower_vals, width,bottom=lower_vals,tick_label=ticks,align="center" , edgecolor="black",linewidth=1.3)
    
    plt.xticks(ind,ticks, rotation='vertical')
    # loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
    # ax.xaxis.set_major_locator(loc)
    # ax.set_ylim([0,1])
    #ax.set_xlim([0,33])
    plt.ylabel('relevance',fontsize=19)
    plt.xlabel('feature',fontsize=19)

    return fig