"""Render the publication-facing two-figure layout from frozen source tables.

No simulator runs, fitting, bootstrap resampling, or new hypothesis tests occur.
Figure 2a waveforms are illustrative; Figures 1 and 2b-c are data-derived.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK="#292827"; RED="#B54A3C"; TAUPE="#756458"; GRAY="#BFBEBB"; PALE="#D8D4CD"

def export(fig, out, name):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=400 if ext=="png" else None, facecolor="white")
    plt.close(fig)

def recovery(src, out):
    summ=pd.read_csv(src/"plot_data/recovery_summary.csv")
    pred=pd.read_csv(src/"data/mechanism_compensation/parameter_predictions.csv")
    variant="scaled_dendrite_v1_explicit_patch"
    summ=summ[summ.variant.eq(variant)]
    pred=pred[pred.variant.eq(variant) & pred.role.eq("test")]
    assert set(pred.groupby("model").size().values)=={10}
    fig,ax=plt.subplots(1,2,figsize=(7.2,3.35),gridspec_kw={"width_ratios":[1.05,1]})
    for a in ax:
        a.spines[["top","right"]].set_visible(False)
        a.tick_params(length=3,width=.6)
    keys=[("training_mean","Training mean"),("cortical","Cortical"),
          ("effective_mode","Effective mode"),("full_population","All populations")]
    for i,(key,label) in enumerate(keys):
        r=summ.loc[summ.model.eq(key)].iloc[0]; y=3-i
        clr=RED if key=="full_population" else (PALE if key=="training_mean" else TAUPE)
        ax[0].plot([r.lower,r.upper],[y,y],color=clr,lw=2)
        ax[0].scatter([r.mean_standardized_mse],[y],s=26,c=clr)
        if key in ("cortical","full_population"):
            ax[0].text(r.mean_standardized_mse+.085,y+.1,f"{r.mean_standardized_mse:.3f}",fontsize=8.2)
    ax[0].set_yticks([3,2,1,0],[label for _,label in keys])
    ax[0].set(xlim=(0,1.7),ylim=(-.55,3.55),xlabel="Held-out standardized squared error")
    ax[0].set_title("Recovery of generating factors",loc="left",pad=12)
    ax[0].text(-.085,1.10,"a",transform=ax[0].transAxes,fontsize=12,weight="bold")
    cort=pred[pred.model.eq("cortical")][["configuration_group","standardized_mse"]].rename(columns={"standardized_mse":"cortical"})
    allpop=pred[pred.model.eq("full_population")][["configuration_group","standardized_mse"]].rename(columns={"standardized_mse":"allpop"})
    pair=cort.merge(allpop,on="configuration_group",validate="one_to_one")
    assert len(pair)==10
    assert abs(pair.cortical.mean()-.3363)<.003 and abs(pair.allpop.mean()-.1689)<.003
    for r in pair.itertuples():
        ax[1].plot([0,1],[r.cortical,r.allpop],color=PALE,lw=1.1)
    ax[1].scatter(np.zeros(10),pair.cortical,s=28,facecolors="white",edgecolors=TAUPE,lw=1)
    ax[1].scatter(np.ones(10),pair.allpop,s=28,c=RED)
    ax[1].scatter([0,1],[pair.cortical.mean(),pair.allpop.mean()],s=46,marker="D",c=[TAUPE,RED],zorder=5)
    ax[1].set_xticks([0,1],["Cortical","All populations"])
    ax[1].set(xlim=(-.3,1.3),ylim=(0,.92),ylabel="Standardized squared error")
    ax[1].set_title("Ten held-out configurations",loc="left",pad=12)
    ax[1].text(-.10,1.10,"b",transform=ax[1].transAxes,fontsize=12,weight="bold")
    fig.subplots_adjust(left=.15,right=.965,top=.83,bottom=.18,wspace=.36)
    export(fig,out,"Figure_1_Recovery")

def pathways(src, out):
    p=pd.read_csv(src/"data/pathway_balance/population_pathway_balance.csv")
    z=p[p["mode"].eq("paired_finite_pulse_response") & p["scope"].eq("all_incoming")]
    assert len(z)==432 and z.population.value_counts().eq(48).all()
    assert (z.absolute_pathway_mass_V_s>0).all() and (z.net_pathway_mass_V_s>0).all()
    means=z.groupby("population").cancellation_for_inference.mean()
    assert abs(means.loc[4]-.413)<.005 and abs(means.loc[6]-.333)<.005
    fig=plt.figure(figsize=(7.4,6.55),facecolor="white")
    grid=fig.add_gridspec(2,2,height_ratios=(1.12,1),left=.067,right=.974,
                          top=.947,bottom=.087,hspace=.27,wspace=.32)
    a=fig.add_subplot(grid[0,:]);b=fig.add_subplot(grid[1,0]);c=fig.add_subplot(grid[1,1])
    a.set(xlim=(0,1),ylim=(0,1));a.axis("off")
    a.text(0,1.06,"a",fontsize=12,weight="bold",clip_on=False)
    a.text(.055,1.06,"Native pathway summation and cancellation",fontsize=10,clip_on=False)
    x=np.linspace(0,1,300)
    g=np.exp(-((x-.48)/.27)**2)
    raw=[.75*g*np.sin(5*np.pi*x+.4),.62*g*np.sin(4*np.pi*x+1.15),
         -.46*g*np.sin(5.5*np.pi*x-.1),
         .25*(1/(1+np.exp(-(x-.2)*38))-1/(1+np.exp(-(x-.8)*38)))]
    carrier=np.exp(-((x-.51)/.23)**2)*np.sin(4*np.pi*x)
    minor=np.exp(-((x-.47)/.28)**2)*np.sin(7*np.pi*x+1)
    response=[.85*carrier+.12*minor,-.69*carrier+.04*minor,
              .53*carrier-.1*minor,-.64*carrier-.08*minor]
    assert max(abs(sum(response)))<.16
    names=["population 1","population 2","population 4","external drive u"]
    vlabels=["$v_1(t)$","$v_2(t)$","$v_4(t)$","$v_u(t)$"]
    dlabels=["$d_1(t)$","$d_2(t)$","$d_4(t)$","$d_u(t)$"]
    def arrow(x0,y0,x1,y1):
        a.annotate("",xy=(x1,y1),xytext=(x0,y0),
                   arrowprops={"arrowstyle":"-|>","color":GRAY,"linewidth":.95,"mutation_scale":6.5})
    def unit(top,waves,labels,is_pulse):
        ys=[top-.084*i for i in range(4)]; yc=np.mean(ys)
        for i,(y,w) in enumerate(zip(ys,waves)):
            a.text(.012,y,names[i],va="center",fontsize=8,color=INK)
            a.scatter([.198],[y],s=53,facecolors="white",edgecolors=INK,lw=1)
            a.plot([.208,.255],[y,y],color=GRAY,lw=1)
            a.plot(.255+.158*x,y+.025*w,color=RED,lw=1.65)
            a.text(.337,y+.029,labels[i],ha="center",va="bottom",fontsize=8.1)
            a.plot([.413,.467],[y,y],color=GRAY,lw=1);arrow(.467,y,.566,yc)
        a.scatter([.592],[yc],s=720,facecolors="white",edgecolors=INK,lw=1.1,zorder=5)
        a.text(.592,yc,r"$\Sigma$",ha="center",va="center",fontsize=20,zorder=6)
        arrow(.622,yc,.690,yc)
        a.plot(.699+.115*x,yc+.025*sum(waves),color=TAUPE,lw=1.8)
        a.text(.758,yc+.05,r"$\Delta V_1(t)$" if is_pulse else r"$V_1(t)$",
               ha="center",va="bottom",fontsize=9.4)
        if is_pulse:
            a.text(.84,yc+.06,r"$\Delta V_1(t)=\sum_j d_j(t)$",fontsize=9.4)
            a.text(.84,yc-.018,r"$C=1-N/A$",fontsize=10)
            a.text(.84,yc-.078,"Pathway cancellation",fontsize=8.2)
        else:a.text(.84,yc,r"$V_1(t)=\sum_j v_j(t)$",fontsize=9.4)
    unit(.875,raw,vlabels,False);unit(.44,response,dlabels,True)
    a.plot([.015,.98],[.513,.513],color="#E6E3E1",lw=.65)
    a.text(.012,.978,"Native modeled voltage",fontsize=8.8,color=TAUPE)
    a.text(.012,.541,"Pulse-minus-control response",fontsize=8.8,color=TAUPE)
    a.text(.985,.017,"Waveforms are schematic",fontsize=7.8,ha="right",color=TAUPE)
    for ax in (b,c):
        ax.spines[["right","top"]].set_visible(False)
        ax.tick_params(length=2.7,width=.65)
    b.text(-.16,1.095,"b",transform=b.transAxes,fontsize=12,weight="bold")
    b.set_title("Absolute versus net response",loc="left",pad=10)
    xx=np.log10(z.absolute_pathway_mass_V_s.to_numpy())
    yy=np.log10(z.net_pathway_mass_V_s.to_numpy())
    lo=min(xx.min(),yy.min())-.1; hi=max(xx.max(),yy.max())+.1
    b.plot([lo,hi],[lo,hi],color=TAUPE,linestyle="--",lw=.8)
    for mask,col,face,label in [(z.population.eq(1),TAUPE,"white","Population 1"),
                                (z.population.ne(1),RED,RED,"Populations 2-9")]:
        d=z[mask]
        b.scatter(np.log10(d.absolute_pathway_mass_V_s),np.log10(d.net_pathway_mass_V_s),
                  s=10,facecolors=face,edgecolors=col,
                  linewidths=.7 if face=="white" else 0,alpha=.7,label=label)
    b.set(xlim=(lo,hi),ylim=(lo,hi),
          xlabel=r"$\log_{10}$ absolute pathway mass (V s)",
          ylabel=r"$\log_{10}$ net response mass (V s)")
    b.legend(frameon=False,loc="upper left",fontsize=7)
    c.text(-.14,1.095,"c",transform=c.transAxes,fontsize=12,weight="bold")
    c.set_title("Population-specific cancellation",loc="left",pad=10)
    for pop in range(1,10):
        vals=z.loc[z.population.eq(pop),"cancellation_for_inference"].to_numpy()
        col=TAUPE if pop==1 else RED
        bp=c.boxplot([vals],positions=[pop],widths=.53,showfliers=False,
                     patch_artist=True,medianprops={"color":col,"linewidth":1.3},
                     boxprops={"color":col},whiskerprops={"color":col},capprops={"color":col})
        bp["boxes"][0].set_facecolor("white")
        c.scatter(pop+np.linspace(-.2,.2,len(vals)),vals,s=5.5,color=col,alpha=.32)
        c.scatter([pop],[vals.mean()],marker="D",s=25,color=col,
                  edgecolor="white",lw=.7,zorder=5)
    c.set(xlim=(.55,9.45),ylim=(0,.67),xlabel="Model population",
          ylabel="Cancellation index, C")
    c.set_xticks(range(1,10))
    export(fig,out,"Figure_2_Integrated")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument("--out",type=Path,default=None)
    opt=parser.parse_args()
    source=opt.repo/"release/science_advances_20261024"
    out=opt.out or Path(__file__).resolve().parents[1]/"figures"
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.8,
                         "pdf.fonttype":42,"svg.fonttype":"none","axes.linewidth":.75})
    recovery(source,out);pathways(source,out)
    print("Replotted saved results and schematic to",out)
    print("No simulator run, new fit, bootstrap or hypothesis test was performed.")
if __name__=="__main__": main()
