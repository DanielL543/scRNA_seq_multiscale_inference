import matplotlib.pyplot as plt
import te_bifurcation as bf
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import mannwhitneyu

# Define the model
model_2pops = {
    'pars':{
        'r1'     : 0.008,
        'r2'     : 0.001,
        'k1'     : 0.04,
        'k2'     : 0.04,
        'K2'     : 100,
        'n2'     : 6,
        'K3'     : 100,
        'n3'     : 6,
        'K4'     : 100,
        'n4'     : 6,
    },
    'vars':{
        'x1' : \
            'r1*x1 - k1/(1+(z1/K3)^n3+(z2/K4)^n4)*x1 + k2/(1+(z1/K2)^n2)*x2',
        'x2' : \
            'r2*x2 + k1/(1+(z1/K3)^n3+(z2/K4)^n4)*x1 - k2/(1+(z1/K2)^n2)*x2',
        'z1' : '10*(x1/(x1+x2) - z1)', # Fraction of x1 population
        'z2' : '10*(x2/(x1+x2) - z2)'  # Fraction of x2 population
    },
'fns': {}, 'aux': [], 'name':'acdc'}
ics_2pops = {'x1': 10.0, 'x2': 0.0, 'z1':0, 'z2':0} # Initial conditions
def norm_m(m):
    """
    Normalize the cell populations to obtain fractions.
    Parameters
    ----------
    m : roadrunnerSimulation
        The simulation object
    Returns
    -------
    t : numpy array
        The time points
    w1 : numpy array
        The fraction of x1 population
    w2 : numpy array
        The fraction of x2 population
    """
    cols = [x[1:-1] for x in m.colnames[1:]]
    i1 = cols.index('x1')+1
    i2 = cols.index('x2')+1
    print(cols, i1, i2)
    t = m[:,0]
    w1 = m[:,i1]/(m[:,i1]+m[:,i2]) #/m[-1,1]
    w2 = m[:,i2]/(m[:,i1]+m[:,i2]) #/m[-1,2]
    return t, w1, w2
def get_t_half(m):
    """
    Calculate the saturation time for a given simulation.
    Parameters
    ----------
    m : roadrunner.SimulationResult
        The simulation result object containing time and population data.
    Returns
    -------
    float
        The time at which the population reaches 95% saturation.
    """
    t, w1, w2 = norm_m(m)
    if w1[0] > w2[0]:
        w = w2
    else:
        w = w1
    return t[np.argmin(abs(w/(w[-1]-w.min()) - 0.95))]
r = bf.model2te(model_2pops, ics=ics_2pops)
m = r.simulate(0, 100, 10000)
def reset_Ks(Ks=None):
    """
    Reset the K parameters of the model.
    Parameters
    ----------
    Ks : list of floats, optional
        The new values of K2, K3 and K4. If not provided, the default values are used.
    """
    if not Ks is None:
        r['K2'], r['K3'], r["K4"] = Ks[0], Ks[1], Ks[2]
    else:
        r['K2'], r['K3'], r["K4"] = 100, 100, 100
def reset_all_pars():
    """
    Reset all parameters of the model to their default values.
    """
    for k in model_2pops["pars"].keys():
        r[k] = model_2pops["pars"][k]


# Show popoulation growth to estimate r1
if 0:
    r.k1 = 0
    r.k2 = 0
    r.reset()
    r.x1, r.x2 = 5, 5
    m = r.simulate(0, 200, 10000)
    reset_all_pars()
    r.reset()
    fig, ax = plt.subplots(figsize=(3, 3))
    fig.subplots_adjust(left=0.2, bottom=0.2)
    ax.plot(m[:,0], m[:,1]+m[:,4], 'b')
    ax.axhline(20, color='gray', ls='--', alpha=0.7)
    ax.axvline(120, color='gray', ls='--', alpha=0.7)
    ax.set_xlim(0, 150)
    ax.set_xlabel('Time (a.u.)')
    ax.set_ylabel('Population size (a.u.)')
    #fig.savefig('./figures/estimate_r1.svg', format='svg', dpi=600)
    plt.show()


################ Scan k2 ########################################
k2s = np.linspace(0.20, 0.000, 100)
def scan_k2(single_Ks=[0.5, 100, 100],
            triple_Ks=[0.5, 0.9, 0.9],
            k2s=np.linspace(0.20, 0.000, 100)):
    i = 0
    w2s, w2cs, ths, thcs = [], [], [], []
    w22s, th22s = [], []
    for k2 in k2s:
        r.reset()
        i += 1
        reset_Ks(single_Ks)
        r['k2'] = k2
        m = r.simulate(0, 400, 1000) ###### Simulate with feedback
        t, w1, w2 = norm_m(m)
        th = get_t_half(m)
        r.reset()
        reset_Ks(triple_Ks)
        m2 = r.simulate(0, 400, 1000) ###### Simulate with triple feedback
        t2, w12, w22 = norm_m(m2)
        th2 = get_t_half(m2)
        #
        r.reset()
        reset_Ks() ############# Remove feedback (Control)
        mc = r.simulate(0, 400, 1000)
        tc, w1c, w2c = norm_m(mc)
        thc = get_t_half(mc)
        w2s.append(w2[-1])
        w2cs.append(w2c[-1])
        #
        ths.append(th)
        thcs.append(thc)
        w22s.append(w22[-1])
        th22s.append(th2)
    return np.array(w2s), np.array(ths), np.array(w2cs), np.array(thcs), np.array(w22s), np.array(th22s)
reset_all_pars()
w2s, ths, w2cs, thcs, w22s, th22s = \
    scan_k2(single_Ks=[0.5, 100, 100],
            triple_Ks=[0.5, 0.9, 0.7], k2s=np.linspace(r.k1*5, 0.000, 100))
fig2, ax2 = plt.subplots(figsize=(3, 3))
fig2.subplots_adjust(left=0.2, bottom=0.2, wspace=0.6)
ax2.scatter(w2cs, thcs, facecolor='w', edgecolors='b', marker='o', s=k2s*500, label='Control')
ax2.scatter(w2s, ths, facecolor='w', edgecolors='r', marker='o', s=k2s*500, label='w/ feedback')
#ax2.scatter(w22s, th22s, facecolor='w', edgecolors='lightgreen', marker='o', s=k2s*500, label='w/ triple feedback')
ax2.set_ylabel('Saturation time (a.u.)')
ax2.set_xlabel(r'Target $X_2$ fraction')
#ax2.set_xlim(0, 1)
ax2.legend()
#fig2.savefig('temp.png')
#fig2.savefig('./figures/scan_k2_2models.svg', format='svg', dpi=600)
#fig2.savefig('./figures/scan_k2_2models_outof50_k1.svg', format='svg', dpi=600)
#
dfper1 = pd.DataFrame({'w2': w2s, 'th': ths, 'Model': ['w/ single\nfeedback']*len(w2s)})  
dfper2 = pd.DataFrame({'w2': w2cs, 'th': thcs, 'Model': ['Control']*len(w2s)})  
dfper3 = pd.DataFrame({'w2': w22s, 'th': th22s, 'Model': ['w/ triple\nfeedback']*len(w2s)})  
df2plot = pd.concat([dfper2, dfper1, dfper3])
fig3, axes = plt.subplot_mosaic("AAA\nAAA\nAAA\nBBB", figsize=(3, 4))
fig3.subplots_adjust(wspace=0.2, hspace=0.2, right=0.85, top=0.85, left=0.15)
ax3, ax4, = axes['A'], axes['B']
fig3.subplots_adjust(left=0.2, bottom=0.2, wspace=0.6)
ax3.scatter(w2cs, thcs, facecolor='w', edgecolors='b', marker='o', s=k2s*500, label='Control')
ax3.scatter(w2s, ths, facecolor='w', edgecolors='r', marker='o', s=k2s*500, label='w/ feedback')
ax3.scatter(w22s, th22s, facecolor='w', edgecolors='lightgreen', marker='o', s=k2s*500, label='w/ triple feedback')
ax3.set_ylabel('Saturation time (a.u.)')
ax3.set_xlabel(r'Target $X_2$ fraction')
ax3.set_ylim(0, 90)
ax3.legend(bbox_to_anchor=(0.05, 1.45), loc='upper left')
sns.kdeplot(data=df2plot, x='w2', hue='Model', palette=['b', 'r', 'lightgreen'], fill=True, ax=ax4, legend=False)
ax4.set_xlim(ax3.get_xlim())
ax4.text(0.59, 0.95, 'SD = '+str(round(dfper2.w2.std()/1, 3)).ljust(5,'0'), color='b', fontsize=10, ha='left', va='top', transform=ax4.transAxes)
ax4.text(0.59, 0.70, 'SD = '+str(round(dfper1.w2.std()/1, 3)).ljust(5,'0'), color='r', fontsize=10, ha='left', va='top', transform=ax4.transAxes)
ax4.text(0.59, 0.45, 'SD = '+str(round(dfper3.w2.std()/1, 3)).ljust(5,'0'), color='g', fontsize=10, ha='left', va='top', transform=ax4.transAxes)
ax3.set_xticklabels([])
ax4.set_ylabel('Density')
ax4.set_xlabel(r'Target $X_2$ fraction')
plt.show()


##################### Plot 3 trajectories with saturation times ####
# Find the k2 values that gives 50% x1. To be used later in two models
i1 = np.searchsorted(w2s, 0.5)
ic = np.searchsorted(w2cs, 0.5)
#print(i1, ic)
#print(w2s[i1], w2cs[ic])
r.reset()
reset_Ks([0.5, 100, 100])
r.k2 = k2s[i1] # Model with feedback. k2 that gives 50% x1
fig, ax = plt.subplots(figsize=(4, 3))
fig.subplots_adjust(left=0.2, bottom=0.2)
m = r.simulate(0, 200, 1000)
t, w1, w2 = norm_m(m)
th = get_t_half(m)
#ax.plot(t, w1, 'b', alpha=0.5)
ax.plot(t, w2, 'r', alpha=0.99, label='w/ feedback')
ax.axvline(th, color='r', ls='--', alpha=0.7)
#
r.reset()
r.k2 = k2s[i1] # Control model. No feedback. Compensated k2
r['K2'] = 10000
m = r.simulate(0, 200, 1000)
t, w1, w2 = norm_m(m)
th = get_t_half(m)
ax.plot(t, w2, 'k', alpha=0.99, label='No feedback')
ax.axvline(th, color='k', ls='--', alpha=0.7)
#
r.reset()
r.k2 = k2s[ic] + 0.0005
r['K2'] = 10000 # Control model. No feedback. 
m = r.simulate(0, 200, 1000)
t, w1, w2 = norm_m(m)
thc = get_t_half(m)
print(thc, th, k2s[ic], k2s[i1])
#ax.plot(t, w1, 'b')
ax.plot(t, w2, 'b', ls='-', alpha=0.9, label='No feedback\nk2 compensated')
ax.axvline(thc, color='b', ls='--', alpha=0.7)
ax.set_xlim(0, 60)
ax.set_xlabel(r'Time (a.u.)')
ax.set_ylabel(r'Fraction of $X_2$ population')
ax.legend()
#fig.savefig('temp.png')
#fig.savefig('./figures/3_trajs.svg',format='svg', dpi=600)
plt.show()



#################### Sensitivity analysis ########################
p2scan = ['r1', 'r2', 'n2', 'n3', 'n4', 'k1']
perfs_all, pss = [], []
for p in p2scan:
    print(p)
    reset_all_pars()
    p_bas = model_2pops['pars'][p]
    if p.startswith('n'):
        ps = [2, 3, 4, 5, 6, 7, 8]
    else:
        ps = np.logspace(np.log10(p_bas*0.1), np.log10(p_bas*10), 15, endpoint=True)
    pss.append(ps)
    print(ps)
    frac = 0.5
    perf, perfc, perf2 = [], [], []
    for pv in ps:
        r[p] = pv
        w2s, ths, w2cs, thcs, w22s, th22s = scan_k2(k2s=np.linspace(r.k1*5, 0.000, 100))
        i1 = np.searchsorted(w2s, frac)
        ic = np.searchsorted(w2cs, frac)
        i2 = np.searchsorted(w22s, frac)
        if i1 > 99 or ic > 99 or i2 > 99 or i1 < 1 or ic < 1 or i2 < 1:
                print('not found')
                perf.append(np.nan)
                perfc.append(np.nan)
                perf2.append(np.nan)
        elif 0:
                perf.append(ths[i1])
                perfc.append(thcs[ic])
                perf2.append(th22s[i2])
        else:
            perf.append(np.std(w2s))
            perfc.append(np.std(w2cs))
            perf2.append(np.std(w22s))
    print(ps, perf, perfc)
    perf_all_p = {'perf': perf, 'perfc': perfc, 'perf2': perf2}
    perfs_all.append(perf_all_p)
    print(ps, perf, perfc)
#
for p, ps, perf_all_p in zip(p2scan, pss, perfs_all):
    perf, perfc, perf2 = perf_all_p['perf'], perf_all_p['perfc'], perf_all_p['perf2']
    fig, ax = plt.subplots(figsize=(3, 3))
    fig.subplots_adjust(left=0.19, bottom=0.16)
    ax.scatter(ps, perfc, facecolors='lightblue', edgecolors='b', marker='o', s=50, label='Control', zorder=3)
    ax.scatter(ps, perf, facecolors='pink', edgecolors='r', marker='o', s=80, label='w/ single feedback', zorder=2)
    ax.scatter(ps, perf2, facecolors='lightgreen', edgecolors='green', marker='o', s=100, label='w/ triple feedback', zorder=1)
    ax.axvline(model_2pops['pars'][p], color='gray', zorder=0, ls='--')
    if np.any(np.isnan(perf)):
        ax.scatter(ps[np.isnan(perf)], np.zeros_like(ps[np.isnan(perf)]), marker='x', s=100, color='k', clip_on=False)
    if not p.startswith('n'):
        ax.set_xscale('log')
    #ax.set_ylabel('Saturation time (a.u.)')
    ax.set_ylabel('Standard deviation')
    ax.set_xlabel(r'Parameter value')
    ax.set_title(r'${}$'.format(p[0]+'_'+p[1]))
    ax.set_ylim(0,)
plt.show()


pK2scan = ['K2', 'K3', 'K4']
perfs_all, pss = [], []
for p in pK2scan:
    print(p)
    reset_all_pars()
    if p == 'K2':
        p_bas = 0.5
        ps = np.logspace(np.log10(p_bas/10**0.5), np.log10(p_bas*10**0.5), 15, endpoint=True)
    if p == 'K3' or p == 'K4':
        p_bas = 0.9
        ps = np.logspace(np.log10(p_bas/10**0.5), np.log10(p_bas*10**0.5), 15, endpoint=True)
    pss.append(ps)
    print(ps)
    frac = 0.5
    perf, perfc, perf2 = [], [], []
    for pv in ps:
        r[p] = pv
        if p == 'K2':
            w2s, ths, w2cs, thcs, w22s, th22s = scan_k2(single_Ks=[pv, 100, 100], triple_Ks=[pv, 0.9, 0.9], k2s=np.linspace(r.k1*5, 0.000, 100))
        elif p == 'K3':
            w2s, ths, w2cs, thcs, w22s, th22s = scan_k2(triple_Ks=[0.5, pv, 0.9], k2s=np.linspace(r.k1*5, 0.000, 100))
        elif p == 'K4':
            w2s, ths, w2cs, thcs, w22s, th22s = scan_k2(triple_Ks=[0.5, 0.9, pv], k2s=np.linspace(r.k1*5, 0.000, 100))
        i1 = np.searchsorted(w2s, frac)
        ic = np.searchsorted(w2cs, frac)
        i2 = np.searchsorted(w22s, frac)
        if i1 > 99 or ic > 99 or i2 > 99 or i1 < 1 or ic < 1 or i2 < 1:
                print('not found')
                perf.append(np.nan)
                perfc.append(np.nan)
                perf2.append(np.nan)
        elif 0:
                perf.append(ths[i1])
                perfc.append(thcs[ic])
                perf2.append(th22s[i2])
        else:
            perf.append(np.std(w2s))
            perfc.append(np.std(w2cs))
            perf2.append(np.std(w22s))
    perf_all_p = {'perf': perf, 'perfc': perfc, 'perf2': perf2}
    perfs_all.append(perf_all_p)
    print(ps, perf, perfc)
for p, ps, perf_all_p in zip(pK2scan, pss, perfs_all):
    perf, perfc, perf2 = perf_all_p['perf'], perf_all_p['perfc'], perf_all_p['perf2']
    fig, ax = plt.subplots(figsize=(3, 3))
    fig.subplots_adjust(left=0.19, bottom=0.16)
    ax.scatter(ps, perfc, facecolors='lightblue', edgecolors='b', marker='o', s=50, label='Control', zorder=3)
    ax.scatter(ps, perf, facecolors='pink', edgecolors='r', marker='o', s=80, label='w/ single feedback', zorder=2)
    ax.scatter(ps, perf2, facecolors='lightgreen', edgecolors='green', marker='o', s=100, label='w/ triple feedback', zorder=1)
    if np.any(np.isnan(perf)):
        ax.scatter(ps[np.isnan(perf)], np.zeros_like(ps[np.isnan(perf)]), marker='x', s=100, color='k', clip_on=False)
    ax.axvline(np.median(ps), color='gray', zorder=0, ls='--')
    if not p.startswith('n'):
        ax.set_xscale('log')
    ax.set_ylabel('Saturation time (a.u.)')
    ax.set_ylabel('Standard deviation')
    ax.set_xlabel(r'Parameter value')
    ax.set_title(r'${}$'.format(p[0]+'_'+p[1]))
    ax.set_ylim(0,)
plt.show()


################### Purterb all parameters ############################
w2s, w2cs, ths, thcs = [], [], [], []
w22s, th22s = [], []
reset_all_pars()
p2exl = ['K2', 'K3', 'K4']
for i in range(1000):
    reset_all_pars()
    r.reset()
    for p in model_2pops["pars"].keys():
        if p not in p2exl:
            p_old = r[p]
            p_min, p_max = p_old * 0.5, p_old * 1.5
            if p.startswith('n'):
                r[p] = int(np.random.uniform(p_min, p_max))
            else:
                r[p] = np.random.uniform(p_min, p_max)
            print(p, p_old, r[p])
    reset_Ks([0.5, 100, 100])
    m = r.simulate(0, 200, 1000) # Single feedback
    t, w1, w2 = norm_m(m)
    th = get_t_half(m)
    w2s.append(w2[-1])
    ths.append(th)
    r.reset()
    reset_Ks([0.5, 0.9, 0.9])
    #reset_Ks([0.5, 0.7, 100])
    m2 = r.simulate(0, 200, 1000) ###### Simulate with triple feedback
    t2, w12, w22 = norm_m(m2)
    th2 = get_t_half(m2)
    reset_Ks([100, 100, 100])
    r.reset()
    m = r.simulate(0, 200, 1000)
    t, w1, w2 = norm_m(m)
    thc = get_t_half(m)
    w2cs.append(w2[-1])
    thcs.append(thc)
    w22s.append(w22[-1])
    th22s.append(th2)
#dfper1 = pd.DataFrame({'w2': w2s, 'th': ths, 'Model': ['w/ single\nfeedback']*len(w2s)})  
dfper1 = pd.DataFrame({'w2': w2s, 'th': ths, 'Model': ['w/ feedback']*len(w2s)})  
dfper2 = pd.DataFrame({'w2': w2cs, 'th': thcs, 'Model': ['Control']*len(w2s)})  
#dfper3 = pd.DataFrame({'w2': w22s, 'th': th22s, 'Model': ['w/ triple\nfeedback']*len(w2s)})  
fig, axes = plt.subplot_mosaic("AAA.\nCCCB\nCCCB\nCCCB", figsize=(4, 4))
fig.subplots_adjust(wspace=0.2, hspace=0.2, right=0.85, top=0.85, left=0.15)
ax1, ax2, ax3 = axes['A'],  axes['B'], axes['C']
df2plot = pd.concat([dfper2, dfper1])
#df2plot = pd.concat([dfper2, dfper1, dfper3])
sns.kdeplot(data=df2plot, x='w2', y='th', hue='Model', palette=['b', 'r', 'lightgreen'], fill=True, ax=ax3) 
sns.scatterplot(data=df2plot, x='w2', y='th', hue='Model', palette=['b', 'r', 'lightgreen'], alpha=0.6, s=10, ax=ax3, legend=False) 
sns.kdeplot(data=df2plot, x='w2', hue='Model', palette=['b', 'r', 'lightgreen'], fill=True, ax=ax1, legend=False) 
sns.kdeplot(data=df2plot, y='th', hue='Model', palette=['b', 'r', 'lightgreen'], fill=True, ax=ax2, legend=False) 
ax1.set_xlim(ax3.get_xlim())
ax2.set_ylim(ax3.get_ylim())
ax1.text(0.02, 0.95, 'SD = '+str(round(dfper2.w2.std()/1, 3)).ljust(5,'0'), color='b', fontsize=10, ha='left', va='top', transform=ax1.transAxes)
ax1.text(0.02, 0.75, 'SD = '+str(round(dfper1.w2.std()/1, 3)).ljust(5,'0'), color='r', fontsize=10, ha='left', va='top', transform=ax1.transAxes)
#ax1.text(0.02, 0.55, 'SD = '+str(round(dfper3.w2.std()/1, 3)).ljust(5,'0'), color='g', fontsize=10, ha='left', va='top', transform=ax1.transAxes)
ax1.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_xticklabels(['0', '0.01'])
ax1.set_xlabel('')
ax2.set_ylabel('')
sns.move_legend(ax3, "lower left", bbox_to_anchor=(1.01, 1.01))
ax3.set_xlabel(r'Target $X_2$ fraction')
ax3.set_ylabel('Saturation time (a.u.)')
plt.show()

for dfp in [dfper1, dfper2, dfper3]:
    print(dfp['Model'].unique()[0], 'W2 STD', dfp['w2'].std(), 'Saturation time', dfp['th'].mean())

print(mannwhitneyu(dfper1['th'], dfper2['th']))
print(mannwhitneyu(dfper1['th'], dfper3['th']))


##### Models with temporal perturbation and re-equilibrium ##################
r.reset()
reset_all_pars()
reset_Ks([0.5, 100, 100])
r.k2 = k2s[i1]
fig, ax = plt.subplots(figsize=(5, 3))
fig.subplots_adjust(left=0.15, bottom=0.2)
m = r.simulate(0, 100, 1000)
t, w1, w2 = norm_m(m)
th = get_t_half(m)
#ax.plot(t, w1, 'b', t, w2, 'r', alpha=0.5)
r.x2 = r.x2/10
mcont = r.simulate(100, 200, 1000)
tcont, w1cont, w2cont = norm_m(mcont)
t = np.concatenate((t, tcont[1:]))
w1 = np.concatenate((w1, w1cont[1:]))
w2 = np.concatenate((w2, w2cont[1:]))
cols = [x[1:-1] for x in mcont.colnames[1:]]
ix1 = cols.index('x1')+1
ix2 = cols.index('x2')+1
x2 = np.concatenate((m[:, ix2], mcont[1:, ix2]))
ax.plot(t, w2, 'r', alpha=0.99)
ax2 = ax.twinx()
ax2.plot(t, x2, 'orange', ls='-', alpha=0.9, label='w/ feedback')
#
r.reset()
r.k2 = k2s[ic] + 0.0005
reset_Ks([1000, 1000, 1000])
m = r.simulate(0, 100, 1000)
t, w1, w2 = norm_m(m)
thc = get_t_half(m)
#print(thc, th, k2s[ic], k2s[i1])
r.x2 = r.x2/10
mcont = r.simulate(100, 200, 1000)
tcont, w1cont, w2cont = norm_m(mcont)
t = np.concatenate((t, tcont[1:]))
w1 = np.concatenate((w1, w1cont[1:]))
w2 = np.concatenate((w2, w2cont[1:]))
x2 = np.concatenate((m[:, ix2], mcont[1:, ix2]))
#ax.plot(t, w1, 'b',)
ax.plot(t, w2, 'b', label='Control')
ax2.plot(t, x2, 'orange', ls='--', alpha=0.9, label='Control')
ax.set_xlim(75, 200)
ax.set_xlabel(r'Time (a.u.)')
ax.set_ylabel(r'Fraction of $X_2$ population')
ax2.spines['right'].set_color('orange')
ax2.yaxis.label.set_color('orange')
ax2.tick_params(axis='y', colors='orange')
ax2.set_ylabel(r'Population size of $X_2$ (a.u.)', color='orange')
ax2.legend()
plt.show()


################ Scan k2 with 0 X1 initial conditions ###############
i = 0
w1s, w1cs, ths, thcs = [], [], [], []
w12s, th12s = [], []
k2s = np.linspace(0.20, 0.001, 100)
r.x1, r.x2 = 0, 10
for k2 in k2s:
    r.reset()
    r.x1, r.x2 = 0, 10
    i += 1
    reset_Ks([0.5, 100, 100])
    r['k2'] = k2
    m = r.simulate(0, 200, 1000) ###### Simulate with feedback
    t, w1, w2 = norm_m(m)
    th = get_t_half(m)
    r.reset()
    r.x1, r.x2 = 0, 10
    reset_Ks([0.5, 0.9, 0.5])
    r['k2'] = k2
    m2 = r.simulate(0, 200, 1000) ###### Simulate with triple feedback
    t2, w12, w22 = norm_m(m2)
    th2 = get_t_half(m2)
    #
    r.reset()
    r.x1, r.x2 = 0, 10
    reset_Ks() ############# Remove feedback (Control)
    r['k2'] = k2
    mc = r.simulate(0, 200, 1000)
    tc, w1c, w2c = norm_m(mc)
    thc = get_t_half(mc)
    #print(f"k2 {k2:.2f} w1 {w1[-1]:.2f} w2 {w2[-1]:.2f} w1c {w1c[-1]:.2f} w2c {w2c[-1]:.2f}")
    w1s.append(w1[-1])
    w1cs.append(w1c[-1])
    ths.append(th)
    thcs.append(thc)
    w12s.append(w12[-1])
    th12s.append(th2)
i1 = np.searchsorted(1-np.array(w1s), 0.5)
ic = np.searchsorted(1-np.array(w1cs), 0.5)
i21 = np.searchsorted(1-np.array(w12s), 0.5)
fig2, ax2 = plt.subplots(figsize=(3, 3))
fig2.subplots_adjust(left=0.2, bottom=0.2, wspace=0.6)
ax2.scatter(w1s, ths, facecolor='w', edgecolors='r', marker='o', s=k2s*500)
ax2.scatter(w1cs, thcs, facecolor='w', edgecolors='b', marker='o', s=k2s*500)
ax2.set_ylabel('Saturation time (a.u.)')
ax2.set_xlabel(r'Target $X_1$ fraction')
ax2.legend()
plt.show()

##################### Plot 3 trajectories with saturation times, with 0 X1 ####
# Find the k2 values that gives 50% x1. To be used later in two models
r.reset()
r.x1, r.x2 = 0, 10
reset_Ks([0.5, 100, 100])
r.k2 = k2s[i1] # Model with feedback. k2 that gives 50% x1
fig, ax = plt.subplots(figsize=(4, 3))
fig.subplots_adjust(left=0.2, bottom=0.2)
m = r.simulate(0, 200, 1000)
t, w1, w2 = norm_m(m)
th = get_t_half(m)
ax.plot(t, w2, 'r', alpha=0.99, label='w/ feedback')
ax.axvline(th, color='r', ls='--', alpha=0.7)
#
r.reset()
reset_Ks()
r.x1, r.x2 = 0, 10
r.k2 = k2s[i1] # Control model. No feedback. Compensated k2
r['K'] = 10000
reset_Ks([0.5, 100, 100])
m = r.simulate(0, 200, 1000)
t, w1, w2 = norm_m(m)
th = get_t_half(m)
ax.plot(t, w2, 'k', alpha=0.99, label='No feedback')
ax.axvline(th, color='k', ls='--', alpha=0.7)
#
r.reset()
r.x1, r.x2 = 0, 10
r.k2 = k2s[ic] + 0.0005
r['K'] = 10000 # Cnontrol model. No feedback. 
m = r.simulate(0, 200, 1000)
t, w1, w2 = norm_m(m)
thc = get_t_half(m)
ax.plot(t, w2, 'b', ls='-', alpha=0.9, label='Double feedback\nk2 compensated')
ax.axvline(thc, color='b', ls='--', alpha=0.7)
#######################################################################
r.reset()
r.x1, r.x2 = 0, 10
r['K'] = model_2pops['pars']['K']
r['K2'] = 0.8
r['K3'] = model_2pops['pars']['K3']
r.k2 = k2s[i21] # Model with feedback. k2 that gives 50% x1
m = r.simulate(0, 200, 1000)
t, w1, w2 = norm_m(m)
th = get_t_half(m)
print(thc, th, k2s[ic], k2s[i1])
ax.plot(t, w2, 'purple', ls='-', alpha=0.9, label='Double feedback\nk2 compensated')
ax.axvline(th, color='purple', ls='--', alpha=0.7)
ax.set_xlim(0, 60)
ax.set_xlabel(r'Time (a.u.)')
ax.set_ylabel(r'Fraction of $X_2$ population')
ax.legend()
plt.show()

