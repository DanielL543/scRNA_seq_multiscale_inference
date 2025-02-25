import matplotlib.pyplot as plt
import te_bifurcation as bf
import numpy as np

model_2pops = {
    'pars':{
        'r1'     : 0.01,
        'r2'     : 0.001,
        'k1'    : 0.04,
        'k2'    : 0.04,
        'K'     : 0.5,
        'Ka'     : 0.5/10000,
        'n'     : 6,
        'na'     : 4,
        'g'     : 10,
        's1'    : 1,
        'k2_basal_frac' : 0.0
    },
    'vars':{
        'x1' : \
            'r1*x1 - k1*x1 + k2*(k2_basal_frac)*x2 + k2*(1-k2_basal_frac)/(1+(z1/K)^n*s1)*((z2/Ka)^na/(1+(z2/Ka)^na))*x2',
        'x2' : \
            'r2*x2 + k1*x1 - k2*(k2_basal_frac)*x2 - k2*(1-k2_basal_frac)/(1+(z1/K)^n*s1)*((z2/Ka)^na/(1+(z2/Ka)^na))*x2',
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


################ Scan k2 ########################################
i = 0
w2s, w2cs, ths, thcs = [], [], [], []
k2s = np.linspace(0.20, 0.000, 100)
a, b = 0, 0
for k2 in k2s:
    r.reset()
    i += 1
    r['K'] = model_2pops['pars']['K']
    #r['K'] = 0.4
    r['k2'] = k2
    r.s1 = 1
    m = r.simulate(0, 200, 1000) ###### Simulate with feedback
    t, w1, w2 = norm_m(m)
    th = get_t_half(m)
    r.reset()
    r['k2'] = k2
    r['K'] = 10000  ############# Remove feedback (Control)
    r.s1 = 1
    mc = r.simulate(0, 200, 1000)
    tc, w1c, w2c = norm_m(mc)
    thc = get_t_half(mc)
    #print(f"k2 {k2:.2f} w1 {w1[-1]:.2f} w2 {w2[-1]:.2f} w1c {w1c[-1]:.2f} w2c {w2c[-1]:.2f}")
    w2s.append(w2[-1])
    w2cs.append(w2c[-1])
    ths.append(th)
    thcs.append(thc)
    if i == 32:
        print(w2c[-1])
        a = w2c[-1]
    elif i == 75:
        print(w2[-1])
        b = w2[-1]
fig2, ax2 = plt.subplots(figsize=(3, 3))
fig2.subplots_adjust(left=0.2, bottom=0.2, wspace=0.6)
ax2.plot(w2s, ths, 'r', ls='-', marker='o', markersize=1, label='w/ feedback')
ax2.plot(w2cs, thcs, 'b', ls='-', marker='o', markersize=1, label='control')
ax2.set_ylabel('Saturation time (a.u.)')
ax2.set_xlabel(r'Target $X_2$ fraction')
fig2.savefig('temp.png')
#plt.show()


##################### Plot 3 trajectories with saturation times ####
# Find the k2 values that gives 50% x1. To be used later in two models
i1 = np.searchsorted(np.array(w2s), 0.5)
ic = np.searchsorted(np.array(w2cs), 0.5)
#print(i1, ic)
#print(w2s[i1], w2cs[ic])
r.reset()
r['K'] = model_2pops['pars']['K']
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
r['K'] = 10000 # Cnontrol model. No feedback. 
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


########################### Models with perturbation ##################
r.reset()
r['K'] = model_2pops['pars']['K']
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
