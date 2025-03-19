import matplotlib.pyplot as plt
import te_bifurcation as bf
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import mannwhitneyu

# Define the model
model_2pops = {
    'pars':{
        'r1'     : 0.01,
        'r2'     : 0.001,
        'k1'    : 0.04,
        'k2'    : 0.04,
        'K'     : 0.5,
        'n'     : 6,
        'K2'     : 100,
        'n2'     : 6,
        'K3'     : 100,
        'n3'     : 6,
        'g'     : 10,
    },
    'vars':{
        'x1' : \
            'r1*x1 - k1/(1+(z1/K2)^n2+(z2/K3)^n3)*x1 + k2/(1+(z1/K)^n)*x2',
        'x2' : \
            'r2*x2 + k1/(1+(z1/K2)^n2+(z2/K3)^n3)*x1 - k2/(1+(z1/K)^n)*x2',
        'z1' : 'g*(x1/(x1+x2) - z1)', # Fraction of x1 population
        'z2' : 'g*(x2/(x1+x2) - z2)'  # Fraction of x2 population
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
r = bf.model2te(model_2pops, ics=ics_2pops)
m = r.simulate(0, 100, 10000)
def reset_Ks(Ks=None):
    """
    Reset the K parameters of the model.
    Parameters
    ----------
    Ks : list of floats, optional
        The new values of K, K2 and K3. If not provided, the default values are used.
    """
    if not Ks is None:
        r['K'], r['K2'], r["K3"] = Ks[0], Ks[1], Ks[2]
    else:
        r['K'], r['K2'], r["K3"] = 100, 100, 100
def reset_all_pars():
    """
    Reset all parameters of the model to their default values.
    """
    for k in model_2pops["pars"].keys():
        r[k] = model_2pops["pars"][k]



################ Scan k2 ########################################
i = 0
w2s, w2cs, ths, thcs = [], [], [], []
w22s, th22s = [], []
k2s = np.linspace(0.20, 0.000, 100)
a, b = 0, 0
for k2 in k2s:
    r.reset()
    i += 1
    #reset_Ks([0.5, 100, 100])
    reset_Ks([0.5, 100, 100])
    r['k2'] = k2
    m = r.simulate(0, 200, 1000) ###### Simulate with feedback
    t, w1, w2 = norm_m(m)
    th = get_t_half(m)
    r.reset()
    #reset_Ks([0.5, 100, 0.5])
    reset_Ks([0.5, 0.7, 100])
    m2 = r.simulate(0, 200, 1000) ###### Simulate with double feedback
    t2, w12, w22 = norm_m(m2)
    th2 = get_t_half(m2)
    #
    r.reset()
    reset_Ks() ############# Remove feedback (Control)
    r['k2'] = k2
    mc = r.simulate(0, 200, 1000)
    tc, w1c, w2c = norm_m(mc)
    thc = get_t_half(mc)
    w2s.append(w2[-1])
    w2cs.append(w2c[-1])
    ths.append(th)
    thcs.append(thc)
    w22s.append(w22[-1])
    th22s.append(th2)
    if i == 32:
        print(w2c[-1])
        a = w2c[-1]
    elif i == 75:
        print(w2[-1])
        b = w2[-1]
fig2, ax2 = plt.subplots(figsize=(3, 3))
fig2.subplots_adjust(left=0.2, bottom=0.2, wspace=0.6)
#ax2.plot(w2s, ths, 'r', ls='-', marker='o', markersize=1, label='w/ feedback')
#ax2.plot(w2cs, thcs, 'b', ls='-', marker='o', markersize=1, label='control')
#ax2.plot(w22s, th22s, 'k', ls='-', marker='o', markersize=1, label='double feedback')
ax2.scatter(w2s, ths, facecolor='w', edgecolors='r', marker='o', s=k2s*500, label='w/ feedback')
ax2.scatter(w2cs, thcs, facecolor='w', edgecolors='b', marker='o', s=k2s*500, label='control')
#ax2.scatter(w22s, th22s, facecolor='w', edgecolors='k', marker='o', s=k2s*500)
ax2.set_ylabel('Saturation time (a.u.)')
ax2.set_xlabel(r'Target $X_2$ fraction')
ax2.legend()
fig2.savefig('temp.png')
#fig2.savefig('./figures/scan_k2_2models.svg', format='svg', dpi=600)
#plt.show()


##################### Plot 3 trajectories with saturation times ####
# Find the k2 values that gives 50% x1. To be used later in two models
i1 = np.searchsorted(np.array(w2s), 0.5)
ic = np.searchsorted(np.array(w2cs), 0.5)
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
r['K'] = 10000
m = r.simulate(0, 200, 1000)
t, w1, w2 = norm_m(m)
th = get_t_half(m)
ax.plot(t, w2, 'k', alpha=0.99, label='No feedback')
ax.axvline(th, color='k', ls='--', alpha=0.7)
#
r.reset()
r.k2 = k2s[ic] + 0.0005
r['K'] = 10000 # Control model. No feedback. 
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
fig.savefig('temp.png')
#plt.show()


################### Purterb all parameters ############################
w2s, w2cs, ths, thcs = [], [], [], []
w22s, th22s = [], []
reset_all_pars()
p2exl = ['g', 'n', 'K', 'n2', 'n3', 'K2', 'K3']
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
    m = r.simulate(0, 200, 1000) # Single feedback
    t, w1, w2 = norm_m(m)
    th = get_t_half(m)
    w2s.append(w2[-1])
    ths.append(th)
    r.reset()
    reset_Ks([0.5, 0.9, 0.5])
    #reset_Ks([0.5, 0.7, 100])
    m2 = r.simulate(0, 200, 1000) ###### Simulate with double feedback
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
dfper1 = pd.DataFrame({'w2': w2s, 'th': ths, 'Model': ['w/ feedback']*len(w2s)})  
dfper2 = pd.DataFrame({'w2': w2cs, 'th': thcs, 'Model': ['Control']*len(w2s)})  
dfper3 = pd.DataFrame({'w2': w22s, 'th': th22s, 'Model': ['w/ triple feedback']*len(w2s)})  
fig, axes = plt.subplot_mosaic("AAA.\nCCCB\nCCCB\nCCCB", figsize=(4, 4))
fig.subplots_adjust(wspace=0.1, hspace=0.1)
ax1, ax2, ax3 = axes['A'],  axes['B'], axes['C']
#df2plot = pd.concat([dfper1, dfper2])
df2plot = pd.concat([dfper1, dfper2, dfper3])
sns.kdeplot(data=df2plot, x='w2', y='th', hue='Model', palette=['r', 'b', 'lightgreen'], fill=True, ax=ax3) 
sns.scatterplot(data=df2plot, x='w2', y='th', hue='Model', palette=['r', 'b', 'lightgreen'], alpha=0.6, s=10, ax=ax3, legend=False) 
sns.kdeplot(data=df2plot, x='w2', hue='Model', palette=['r', 'b', 'lightgreen'], fill=True, ax=ax1, legend=False) 
sns.kdeplot(data=df2plot, y='th', hue='Model', palette=['r', 'b', 'lightgreen'], fill=True, ax=ax2, legend=False) 
ax1.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_xticklabels(['0', '0.01'])
ax1.set_xlabel('')
ax2.set_ylabel('')
ax1.set_xlim(ax3.get_xlim())
ax2.set_ylim(ax3.get_ylim())
#sns.move_legend(ax3, "upper right", bbox_to_anchor=(1.65, 1.35))
sns.move_legend(ax3, "lower left", bbox_to_anchor=(1.01, 1.01))
ax3.set_xlabel(r'Target $X_2$ fraction')
ax3.set_ylabel('Saturation time (a.u.)')
fig.savefig('temp.png')
#fig.savefig('./figures/per2d3models.svg', format='svg', dpi=600)

for dfp in [dfper1, dfper2, dfper3]:
    print(dfp['Model'].unique()[0], 'W2 STD', dfp['w2'].std(), 'Saturation time', dfp['th'].mean())

print(mannwhitneyu(dfper1['th'], dfper2['th']))
print(mannwhitneyu(dfper1['th'], dfper3['th']))


##### Models with temporal perturbation and re-equilibrium ##################
r.reset()
reset_all_pars()
reset_Ks()
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
x2 = np.concatenate((m[:, 2], mcont[1:, 2]))
#ax.plot(t, w1, 'b', 
ax.plot(t, w2, 'r', alpha=0.99)
ax2 = ax.twinx()
ax2.plot(t, x2, 'orange', ls='-', alpha=0.9, label='w/ feedback')
#
r.reset()
r.k2 = k2s[ic] + 0.0005
r['K'] = 10000
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
x2 = np.concatenate((m[:, 2], mcont[1:, 2]))
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
fig.savefig('temp.png')
#plt.show()


################ Scan k2 with 0 X1 initial conditions ###############
i = 0
w1s, w1cs, ths, thcs = [], [], [], []
w12s, th12s = [], []
k2s = np.linspace(0.30, 0.01, 100)
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
    reset_Ks([0.5, 100, 0.5])
    r['k2'] = k2
    m2 = r.simulate(0, 200, 1000) ###### Simulate with double feedback
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
fig2, ax2 = plt.subplots(figsize=(3, 3))
fig2.subplots_adjust(left=0.2, bottom=0.2, wspace=0.6)
#ax2.plot(w1s, ths, 'r', ls='-', marker='o', markersize=1, label='w/ feedback')
#ax2.plot(w1cs, thcs, 'b', ls='-', marker='o', markersize=1, label='control')
#ax2.plot(w12s, th12s, 'k', ls='-', label='double feedback')
ax2.scatter(w1s, ths, facecolor='w', edgecolors='r', marker='o', s=k2s*500)
ax2.scatter(w1cs, thcs, facecolor='w', edgecolors='b', marker='o', s=k2s*500)
#ax2.scatter(w12s, th12s, facecolor='w', edgecolors='k', marker='o', s=k2s*500)
ax2.set_ylabel('Saturation time (a.u.)')
ax2.set_xlabel(r'Target $X_1$ fraction')
ax2.legend()
fig2.savefig('temp.png')
#fig2.savefig('./figures/scan_k2_with_0_x1.svg', format='svg', dpi=600)
#plt.show()

##################### Plot 3 trajectories with saturation times, with 0 X1 ####
# Find the k2 values that gives 50% x1. To be used later in two models
i1 = np.searchsorted(1-np.array(w1s), 0.5)
ic = np.searchsorted(1-np.array(w1cs), 0.5)
i21 = np.searchsorted(1-np.array(w12s), 0.5)
#print(i1, ic)
#print(w2s[i1], w2cs[ic])
r.reset()
r.x1, r.x2 = 0, 10
r['K'] = model_2pops['pars']['K']
r['K2'] = model_2pops['pars']['K2']
r['K3'] = model_2pops['pars']['K3']
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
r.x1, r.x2 = 0, 10
r.k2 = k2s[i1] # Control model. No feedback. Compensated k2
r['K'] = 10000
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
#fig, ax = plt.subplots(figsize=(4, 3))
#fig.subplots_adjust(left=0.2, bottom=0.2)
m = r.simulate(0, 200, 1000)
t, w1, w2 = norm_m(m)
th = get_t_half(m)
#ax.plot(t, w1, 'b', alpha=0.5)
#ax.plot(t, w2, 'r', alpha=0.99, label='w/ feedback')
#ax.axvline(th, color='r', ls='--', alpha=0.7)
print(thc, th, k2s[ic], k2s[i1])
#ax.plot(t, w1, 'b')
ax.plot(t, w2, 'purple', ls='-', alpha=0.9, label='Double feedback\nk2 compensated')
ax.axvline(th, color='purple', ls='--', alpha=0.7)
ax.set_xlim(0, 60)
ax.set_xlabel(r'Time (a.u.)')
ax.set_ylabel(r'Fraction of $X_2$ population')
ax.legend()
fig.savefig('temp.png')
#plt.show()