import matplotlib.pyplot as plt
import te_bifurcation as bf
import numpy as np
import seaborn as sns
import pandas as pd

# Define the model
model_3pops = {
    'pars':{
        'r1'     : 0.008,
        'r2'     : 0.001,
        'r3'     : 0.002,
        'k1'     : 0.04,
        'k2'     : 0.04,
        'k3'     : 0.04,
        'k4'     : 0.04,
        'k5'     : 0.04,
        'k6'     : 0.04,
        'K2'     : 100,
        'n2'     : 6,
        'K3'     : 100,
        'n3'     : 6,
        'K4'     : 100,
        'n4'     : 6,
        'K5'     : 100,
        'n5'     : 6,
        'K6'     : 100,
        'n6'     : 6,
    },
    'vars':{
        'x1' : \
            'r1*x1 - k1/(1+(z1/K3)^n3+(z2/K4)^n4+(z3/K5)^n5)*x1 + k2/(1+(z1/K2)^n2+(z3/K6)^n6)*x2 + k3*x3 - k5/(1+(z3/K5)^n5)*x1',
        'x2' : \
            'r2*x2 + k1/(1+(z1/K3)^n3+(z2/K4)^n4+(z3/K5)^n5)*x1 - k2/(1+(z1/K2)^n2+(z3/K6)^n6)*x2 + k4*x3 - k6/(1+(z3/K6)^n6)*x2',
        'x3' : \
            'r3*x3 - k3*x3 - k4*x3 + k5/(1+(z3/K5)^n5)*x1 + k6/(1+(z3/K6)^n6)*x2',
        'z1' : '10*(x1/(x1+x2+x3) - z1)', # Fraction of x1 population
        'z2' : '10*(x2/(x1+x2+x3) - z2)',  # Fraction of x2 population
        'z3' : '10*(x3/(x1+x2+x3) - z3)'  # Fraction of x3 population
    },
'fns': {}, 'aux': [], 'name':'acdc'}
ics_3pops = {'x1': 10.0, 'x2': 0.0, 'x3':0.0, 'z1':0, 'z2':0, 'z3':0} # Initial conditions
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
    i3 = cols.index('x3')+1
    print(cols, i1, i2, i3)
    t = m[:,0]
    w1 = m[:,i1]/(m[:,i1]+m[:,i2]+m[:,i3])
    w2 = m[:,i2]/(m[:,i1]+m[:,i2]+m[:,i3])
    w3 = m[:,i3]/(m[:,i1]+m[:,i2]+m[:,i3])
    return t, w1, w2
def get_t_half(m):
    """
    Calculate the half-saturation time for a given simulation.
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
r = bf.model2te(model_3pops, ics=ics_3pops)
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
        r['K2'], r['K3'], r["K4"], r["K5"], r["K6"] = Ks[0], Ks[1], Ks[2], Ks[3], Ks[4]
    else:
        r['K2'], r['K3'], r["K4"], r["K5"], r["K6"] = 1000, 1000, 1000, 1000, 1000
def reset_all_pars():
    """
    Reset all parameters of the model to their default values.
    """
    for k in model_3pops["pars"].keys():
        r[k] = model_3pops["pars"][k]


################ Scan k2 ########################################
k2s = np.linspace(0.20, 0.000, 100)
def scan_k2(Kset_1=[0.5, 100, 100, 100, 100],
            Kset_2=[0.5, 0.9, 0.9, 100, 100],
            Kset_3=[0.5, 100, 100, 0.9, 0.9],
            Kset_4=[0.5, 0.9, 0.9, 0.9, 0.9],
            k2s=np.linspace(0.20, 0.000, 100)):
    i = 0
    w2s, w2cs, ths, thcs = [], [], [], []
    w22s, th22s = [], []
    w23s, th23s = [], []
    w24s, th24s = [], []
    for k2 in k2s:
        r.reset()
        i += 1
        reset_Ks(Kset_1)
        r['k2'] = k2
        m = r.simulate(0, 400, 1000) ###### Simulate with Kset_1
        t, w1, w2 = norm_m(m)
        th = get_t_half(m)
        r.reset()
        reset_Ks(Kset_2)
        m2 = r.simulate(0, 400, 1000) ###### Simulate with Kset_2
        t2, w12, w22 = norm_m(m2)
        th2 = get_t_half(m2)
        #
        r.reset()
        reset_Ks(Kset_3)
        m3 = r.simulate(0, 400, 1000) ###### Simulate with Kset_3
        t3, w13, w23 = norm_m(m3)
        th3 = get_t_half(m3)
        #
        r.reset()
        reset_Ks(Kset_4)
        m4 = r.simulate(0, 400, 1000) ###### Simulate with Kset_4
        t4, w14, w24 = norm_m(m4)
        th4 = get_t_half(m4)
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
        w23s.append(w23[-1])
        th23s.append(th3)
        w24s.append(w24[-1])
        th24s.append(th4)
    reset_all_pars()
    return np.array(w2s), np.array(ths), np.array(w2cs), np.array(thcs), np.array(w22s), np.array(th22s), np.array(w23s), np.array(th23s), np.array(w24s), np.array(th24s)
reset_all_pars()
#r.r1 = r.r1*11
#r.k1 = r.k1/10
w2s, ths, w2cs, thcs, w22s, th22s, w23s, th23s, w24s, th24s = \
    scan_k2(Kset_1=[0.5, 100, 100, 100, 100],
            Kset_2=[0.5, 100, 100, 0.5, 0.5],
            Kset_3=[100, 100, 100, 0.5, 100],
            Kset_4=[100, 100, 100, 100, 0.5], k2s=np.linspace(r.k1*5, 0.000, 100))
fig2, ax2 = plt.subplots(figsize=(3, 3))
fig2.subplots_adjust(left=0.2, bottom=0.2, wspace=0.6)
ax2.scatter(w2cs, thcs, facecolor='w', edgecolors='b', marker='o', s=k2s*500, label='No feedback')
ax2.scatter(w2s, ths, facecolor='w', edgecolors='r', marker='o', s=k2s*500, label=r'w/ $K_2$ feedback')
ax2.scatter(w22s, th22s, facecolor='w', edgecolors='cyan', marker='o', s=k2s*500, label=r'w/ $K_2$,$K_5$,$K_6$ feedback')
ax2.scatter(w23s, th23s, facecolor='w', edgecolors='m', marker='o', s=k2s*500, label=r'w/ $K_5$ feedback')
ax2.scatter(w24s, th24s, facecolor='w', edgecolors='brown', marker='o', s=k2s*500, label=r'w/ $K_6$ feedback')
ax2.set_ylabel('Saturation time (a.u.)')
ax2.set_xlabel(r'Target $X_2$ fraction')
ax2.legend()
plt.show()

dfper1 = pd.DataFrame({'w2': w2s, 'th': ths, 'Model': ['w/ single\nfeedback']*len(w2s)})  
dfper2 = pd.DataFrame({'w2': w2cs, 'th': thcs, 'Model': ['Control']*len(w2s)})  
dfper3 = pd.DataFrame({'w2': w22s, 'th': th22s, 'Model': ['w/ triple\nfeedback']*len(w2s)})  
dfper4 = pd.DataFrame({'w2': w23s, 'th': th23s, 'Model': ['w/ tri2\nfeedback']*len(w2s)})
dfper5 = pd.DataFrame({'w2': w24s, 'th': th24s, 'Model': ['w/ quad\nfeedback']*len(w2s)})
df2plot = pd.concat([dfper2, dfper1, dfper3, dfper4, dfper5])
fig3, axes = plt.subplot_mosaic("AAA\nAAA\nAAA\nBBB", figsize=(3, 4))
fig3.subplots_adjust(wspace=0.2, hspace=0.2, right=0.85, top=0.85, left=0.15)
ax3, ax4, = axes['A'], axes['B']
fig3.subplots_adjust(left=0.2, bottom=0.2, wspace=0.6)
ax3.scatter(w2cs, thcs, facecolor='w', edgecolors='b', marker='o', s=k2s*500, label='Control')
ax3.scatter(w2s, ths, facecolor='w', edgecolors='r', marker='o', s=k2s*500, label='w/ feedback')
ax3.scatter(w22s, th22s, facecolor='w', edgecolors='lightgreen', marker='o', s=k2s*500, label='w/ triple feedback')
ax3.scatter(w23s, th23s, facecolor='w', edgecolors='m', marker='o', s=k2s*500, label='w/ tri2 feedback')
ax3.scatter(w24s, th24s, facecolor='w', edgecolors='brown', marker='o', s=k2s*500, label='w/ quad feedback')
ax3.set_ylabel('Saturation time (a.u.)')
ax3.set_xlabel(r'Target $X_2$ fraction')
ax3.set_ylim(0, 40)
ax3.legend(bbox_to_anchor=(0.05, 1.45), loc='upper left')
sns.kdeplot(data=df2plot, x='w2', hue='Model', palette=['b', 'r', 'lightgreen', 'm', 'brown'], fill=True, ax=ax4, legend=False)
ax4.set_xlim(ax3.get_xlim())
ax4.text(0.69, 0.99, 'SD = '+str(round(dfper2.w2.std()/1, 3)).ljust(5,'0'), color='b', fontsize=7, ha='left', va='top', transform=ax4.transAxes)
ax4.text(0.69, 0.79, 'SD = '+str(round(dfper1.w2.std()/1, 3)).ljust(5,'0'), color='r', fontsize=7, ha='left', va='top', transform=ax4.transAxes)
ax4.text(0.69, 0.59, 'SD = '+str(round(dfper3.w2.std()/1, 3)).ljust(5,'0'), color='g', fontsize=7, ha='left', va='top', transform=ax4.transAxes)
ax4.text(0.69, 0.39, 'SD = '+str(round(dfper4.w2.std()/1, 3)).ljust(5,'0'), color='m', fontsize=7, ha='left', va='top', transform=ax4.transAxes)
ax4.text(0.69, 0.19, 'SD = '+str(round(dfper5.w2.std()/1, 3)).ljust(5,'0'), color='brown', fontsize=7, ha='left', va='top', transform=ax4.transAxes)
# remove top and right splines for ax4
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax3.set_xticklabels([])
ax4.set_ylabel('Density')
ax4.set_xlabel(r'Target $X_2$ fraction')
plt.show()



################### Purterb all parameters ############################
w2s, w2cs, ths, thcs = [], [], [], []
w22s, th22s = [], []
w23s, th23s = [], []
w24s, th24s = [], []
reset_all_pars()
p2exl = ['K2', 'K3', 'K4', 'K5', 'K6']
for i in range(1000):
    Kset_1=[0.5, 100, 100, 100, 100]
    Kset_2=[0.5, 0.9, 0.9, 100, 100]
    Kset_3=[100, 100, 100, 0.2, 100]
    Kset_4=[100, 100, 100, 100, 0.2]
    reset_all_pars()
    r.reset()
    for p in model_3pops["pars"].keys():
        if p not in p2exl:
            p_old = r[p]
            p_min, p_max = p_old * 0.5, p_old * 1.5
            if p.startswith('n'):
                r[p] = int(np.random.uniform(p_min, p_max))
            else:
                r[p] = np.random.uniform(p_min, p_max)
            print(p, p_old, r[p])
    reset_Ks(Kset_1)
    m = r.simulate(0, 200, 1000) # Simulate with Kset_1
    t, w1, w2 = norm_m(m)
    th = get_t_half(m)
    w2s.append(w2[-1])
    ths.append(th)
    r.reset()
    reset_Ks(Kset_2)
    m2 = r.simulate(0, 200, 1000) ###### Simulate with Kset_2
    t2, w12, w22 = norm_m(m2)
    th2 = get_t_half(m2)
    reset_Ks(Kset_3)
    r.reset()
    m3 = r.simulate(0, 200, 1000) ###### Simulate with Kset_3
    t3, w13, w23 = norm_m(m3)
    th3 = get_t_half(m3)
    reset_Ks(Kset_4)
    r.reset()
    m4 = r.simulate(0, 200, 1000) ###### Simulate with Kset_4
    t4, w14, w24 = norm_m(m4)
    th4 = get_t_half(m4)
    reset_Ks([100, 100, 100, 100, 100])
    r.reset()
    m = r.simulate(0, 200, 1000)
    t, w1, w2 = norm_m(m)
    thc = get_t_half(m)
    w2cs.append(w2[-1])
    thcs.append(thc)
    w22s.append(w22[-1])
    th22s.append(th2)
    w23s.append(w23[-1])
    th23s.append(th3)
    w24s.append(w24[-1])
    th24s.append(th4)
dfper1 = pd.DataFrame({'w2': w2s, 'th': ths, 'Model': [r'w/ $K_2$ feedback']*len(w2s)})
dfper2 = pd.DataFrame({'w2': w2cs, 'th': thcs, 'Model': [r'No feedback']*len(w2s)})
dfper3 = pd.DataFrame({'w2': w22s, 'th': th22s, 'Model': [r'w/ $K_2$, $K_5$, $K_6$ feedback']*len(w2s)})
dfper4 = pd.DataFrame({'w2': w23s, 'th': th23s, 'Model': [r'w/ $K_5$ feedback']*len(w2s)})
dfper5 = pd.DataFrame({'w2': w24s, 'th': th24s, 'Model': [r'w/ $K_6$ feedback']*len(w2s)})
fig, axes = plt.subplot_mosaic("AAA.\nCCCB\nCCCB\nCCCB", figsize=(4, 4))
fig.subplots_adjust(wspace=0.2, hspace=0.2, right=0.85, top=0.85, left=0.15)
ax1, ax2, ax3 = axes['A'],  axes['B'], axes['C']
df2plot = pd.concat([dfper2, dfper1])
df2plot = pd.concat([dfper2, dfper1, dfper3, dfper4, dfper5])
pal = sns.color_palette(['b', 'r', 'cyan', 'm', 'brown'])
sns.kdeplot(data=df2plot, x='w2', y='th', hue='Model', palette=pal, fill=True, ax=ax3) 
sns.scatterplot(data=df2plot, x='w2', y='th', hue='Model', palette=pal, alpha=0.6, s=10, ax=ax3, legend=False) 
sns.kdeplot(data=df2plot, x='w2', hue='Model', palette=pal, fill=True, ax=ax1, legend=False) 
sns.kdeplot(data=df2plot, y='th', hue='Model', palette=pal, fill=True, ax=ax2, legend=False) 
ax1.set_xlim(ax3.get_xlim())
ax2.set_ylim(ax3.get_ylim())
ax1.text(0.02, 1.65, 'SD = '+str(round(dfper2.w2.std()/1, 3)).ljust(5,'0'), color='b', fontsize=10, ha='left', va='top', transform=ax1.transAxes)
ax1.text(0.02, 1.45, 'SD = '+str(round(dfper1.w2.std()/1, 3)).ljust(5,'0'), color='r', fontsize=10, ha='left', va='top', transform=ax1.transAxes)
ax1.text(0.02, 1.25, 'SD = '+str(round(dfper3.w2.std()/1, 3)).ljust(5,'0'), color='cyan', fontsize=10, ha='left', va='top', transform=ax1.transAxes)
ax1.text(0.52, 1.45, 'SD = '+str(round(dfper4.w2.std()/1, 3)).ljust(5,'0'), color='m', fontsize=10, ha='left', va='top', transform=ax1.transAxes)
ax1.text(0.52, 1.25, 'SD = '+str(round(dfper5.w2.std()/1, 3)).ljust(5,'0'), color='brown', fontsize=10, ha='left', va='top', transform=ax1.transAxes)
ax1.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_xticklabels(['0', '0.01'])
ax1.set_xlabel('')
ax2.set_ylabel('')
sns.move_legend(ax3, "lower left", bbox_to_anchor=(1.01, 1.01))
ax3.set_xlabel(r'Target $X_2$ fraction')
ax3.set_ylabel('Saturation time (a.u.)')
plt.show()

