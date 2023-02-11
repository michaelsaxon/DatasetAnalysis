'''
Intended for use on my local laptop just to produce my scatter plots.
Generates beautiful figures.
'''
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd

'''
SOTA,R,BC,M,Delta,%RS,%RR,PECOxH,LBA,RBA,TRNBA
'''

def predband(x, xd, yd, p, func, conf=0.95):
    # x = requested points
    # xd = x data
    # yd = y data
    # p = parameters
    # func = function name
    alpha = 1.0 - conf    # significance
    N = xd.size          # data sample size
    var_n = len(p)  # number of parameters
    # Quantile of Student's t distribution for p=(1-alpha/2)
    q = stats.t.ppf(1.0 - alpha / 2.0, N - var_n)
    # Stdev of an individual measurement
    se = np.sqrt(1. / (N - var_n) * \
                 np.sum((yd - func(xd, *p)) ** 2))
    # Auxiliary definitions
    sx = (x - xd.mean()) ** 2
    sxd = np.sum((xd - xd.mean()) ** 2)
    # Predicted values (best-fit model)
    yp = func(x, *p)
    # Prediction band
    dy = q * se * np.sqrt(1.0+ (1.0/N) + (sx/sxd))
    # Upper & lower prediction bands.
    lpb, upb = yp - dy, yp + dy
    return lpb, upb


def bivariate(x, y, xscale, yscale, xlabel, ylabel, draw_trend = True, trend_bg_color="#E5FFE5", point_color="#005500", yeqx = False):
    plt.rcParams["figure.figsize"] = (4,3)
    if draw_trend:
        # get best fit
        m, b, pcc, pval, stderr = stats.linregress(x,y)
        conf_int  = 2.58*stderr
        lx = np.array(xscale)
        ly = np.array([b + m * xscale[0], b + m * xscale[1]])
        func = lambda x, b, m: b + m * x
        lpb, upb = predband(np.linspace(xscale[0], xscale[1], 100), x, y, (b, m), func)
        lpb = np.maximum(lpb, yscale[0])
        upb = np.minimum(upb, yscale[1])
        plt.fill_between(np.linspace(xscale[0], xscale[1], 100), lpb, upb, color=trend_bg_color)
        plt.plot(lx,ly, color="black", label=f"PCC: {pcc:.3f}", linestyle='dotted')
        print(pcc)
        if yeqx:
            plt.plot(lx,lx, color="red")
    plt.scatter(x,y, c=point_color, marker="+", s=15)
    #plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.ylim(SCALES[ycol])
    #plt.scatter(np.array(xscale), np.array(yscale), alpha = 0, color="#FFFFFF")
    plt.xlim(xscale)
    plt.ylim(yscale)
    plt.subplots_adjust(left=0.165, bottom=0.162, right=0.97, top=0.97, wspace=None, hspace=None)
    plt.show()
    #plt.savefig(f"figs/{setname}/{xcol}_{ycol}.png")
    #plt.savefig(f"figs/{setname}/{xcol}_{ycol}.pdf")
    #plt.clf()
    #print(f"{xcol},{ycol},{pcc},{pval}")


def regions_plot(x, y, marker, xscale, yscale, xlabel, ylabel):
    plt.rcParams["figure.figsize"] = (4,3)
    plt.plot([x[11],x[12]],[y[11],y[12]], c="black", linestyle="dotted", linewidth=1)
    plt.scatter((x[11]+x[12])/2,(y[11]+y[12])/2, c="black", marker="|", edgecolors="black", s=15)
    plt.scatter((x[13]+x[14])/2,(y[13]+y[14])/2, c="black", marker="|", edgecolors="black", s=15)
    plt.scatter((x[15]+x[16])/2,(y[15]+y[16])/2, c="black", marker="|", edgecolors="black", s=15)
    plt.plot([x[13],x[14]],[y[13],y[14]], c="black", linestyle="dotted", linewidth=1)
    plt.plot([x[15],x[16]],[y[15],y[16]], c="black", linestyle="dotted", linewidth=1)
    for i in range(len(marker)):
        plt.scatter(x[i],y[i], c="none", marker=marker[i], s=15, edgecolors="black")
    #plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.ylim(SCALES[ycol])
    plt.plot()
    plt.scatter(np.array(xscale), np.array(yscale), alpha = 0, color="#FFFFFF")
    plt.arrow(55, 37, 28, 0, width=6, head_width = 10, head_length = 10, length_includes_head = True, color="#FFEEEE")
    plt.arrow(86, 42, 0, 50, width=3.5, head_width=7, head_length=15, length_includes_head = True, color="#EEEEFF")
    circle_x = np.linspace(15,49,100)
    circle_center = circle_x.mean()
    circle_rad = circle_center - circle_x.min()
    circle_y = np.sin(np.arccos((circle_x  - circle_center) / circle_rad))
    circle_center_y = 30
    circle_y_rad = 25
    circle_top = circle_center_y + circle_y_rad * circle_y
    circle_bottom = circle_center_y - circle_y_rad * circle_y
    plt.fill_between(circle_x, circle_top, circle_bottom, color="#FEF8CA")
    plt.text(38, 38, "Optimal\nbenchmark\nzone" ,color="#AA8505", weight="bold", rotation=28, ha="center", va="center")
    plt.text(67, 37, "More $\\bf{biased}$" ,color="#CC0000", weight="medium", ha="center", va="center")
    plt.text(86, 65, "More $\\bf{saturated}$" ,color="#0000CC", weight="medium", ha="center", va="center", rotation=90)
    plt.xlim(xscale)
    plt.ylim(yscale)
    plt.subplots_adjust(left=0.165, bottom=0.162, right=0.97, top=0.97, wspace=None, hspace=None)
    plt.show()

# plots to do 
# - to show the sweet spot we're targeting:
# BC vs R (no trendline) 30,80; 30,100 (black)
# - to show that we are reasonably approximating SOTA results in our replication
# SOTA vs R 40,100;40,100 (with additional y=x line?) DARK PURPLE
# - to show PECO works (is predictive of correlation b/w ):
# PECOxH vs %RR DARK BLUE (light blue bg)
# - to show that "similar reasoning" correlates well with bias condition performance
# TRNBA vs BC ORANGE
# - to show that output agreement correlates with recovery rate:
# RBA vs %RR GREEN


csv= pd.read_csv("clusterbias_results.csv")
regions_plot(csv["BC"].to_numpy(), csv["R"].to_numpy(), list(csv["marker"]), [30,90], [30,100], "SSC Accuracy", "PSC accuracy")
#bivariate(csv["SOTA"].to_numpy(), csv["R"].to_numpy(), [40,100], [40,100], "SOTA Accuracy", "Replication accuracy", trend_bg_color="#F5CCF5", point_color="#280028", yeqx = True)
bivariate(csv["PECOxH"].to_numpy()*100, csv["%RR"].to_numpy(), [0,22], [0,100], "PECO score", "$\%R_R$", trend_bg_color="#E5E5FF", point_color="#000055")
bivariate(csv["TRNBA"].to_numpy(), csv["BC"].to_numpy(), [20,90], [20,90], "Replication-Biased token sig. agreement", "SSC accuracy", trend_bg_color="#ffdead", point_color="#cb410b")
bivariate(csv["RBA"].to_numpy(), csv["%RR"].to_numpy(), [20,90], [0,100], "SSC-PSC Output Agreement (NBA)", "$\%R_R$")