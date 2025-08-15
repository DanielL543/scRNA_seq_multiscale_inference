import tellurium as te
import matplotlib.pyplot as plt
import re
import os
import sympy
import numpy as np
import pandas as pd
#from PyDSTool import *
from sys import exit
import matplotlib.patches as patches
from scipy.signal import argrelextrema
from sys import exit

def extract_data(r=None):
    '''
    Extract continuation data from fort.7 and fort.8 files in the current folder after an AUTO run
    r: tellurium object for obtaining variable names via r.fs()
    return:
        data: continuation curve in numpy array (r is not supplied) or pandas dataframe (r is supplied)
        bounds: indices of special points
        boundsh: indices of Hopf bifurcation points
    '''
    with open('fort.7', 'r') as f:
        lines = f.readlines()

    data = []
    for line in lines[12:]:
        if re.match(r'\s+0', line):
            break
        l = line.rstrip()
        fbs = [0, 3, 9, 13, 19]
        fs = []
        for w in range(1, len(fbs)):
            if l[w] != '-':
                fs.append(l[fbs[w-1]+1:fbs[w]+1])
            else:
                fs.append(l[fbs[w-1]+1:fbs[w]])
        fs.extend(re.split(r' +', l.strip()[17:]))
        data.append([float(f) for f in fs if f != ''])
    data = np.array(data)
    if len(data.shape) == 1:
        return [], [], []
    data = data[data[:,3]>0,:]
    idsn = np.where(data[:,1]<0)[0]
    idsp = np.where(data[:,1]>0)[0]

    bksn = np.where((idsn[1:]-idsn[:-1])>1)[0]
    bksp = np.where((idsp[1:]-idsp[:-1])>1)[0]
    bks = np.where((data[:,2]==2)|(data[:,2]==1))[0]
    hpts = np.where((data[:,2]==3))[0]

    bounds = [0]+list(bks)+[len(data)]

    boundsh = hpts

    with open('fort.8', 'r') as f:
        f_str = f.read()

    blks = re.split('\n +[12] +.*\n', f_str)
    half_blk = int(blks[0].count('\n')/2)
    numlines = [re.split("\n",blk) for blk in blks]
    numlines[0] = numlines[0][1:]
    states = [[float(num)
        for num in re.split(' +', "".join(lines[:]).strip())[1:]
        ] for lines in numlines]
    data8 = np.array(states)[:,:-20]

    if data8.shape[1] > data.shape[1]-6:
        if data.shape[0] != data8[:,data.shape[1]-6:].shape[0]:
            return [], [], []
        data = np.hstack([data, data8[:,data.shape[1]-6:]])

    if r:
        data = pd.DataFrame(data=data, columns=['ST', 'PT', 'TY', 'LAB', 'PAR', 'L2-NORM']+r.fs())

    return data, bounds, boundsh

def find_segs_from_bif(data=None):
    '''
    Partition AUTO-generated continuation data into segments (branches)
    Return: stabs: stability of each branch
        starts: start index of each branch
        stops: stop index of each branch
    '''
    if isinstance(data, np.ndarray):
        stabs = np.sign(data[:,1])
    else:
        stabs = np.sign(data.values[:,1])
    starts = np.where(abs(stabs[1:] - stabs[:-1])==2)[0]
    #plt.plot(stabs[1:] - stabs[:-1])
    #plt.show()
    #print(starts)
    starts[1:] += 1
    stops = np.append(starts[1:], len(stabs)-1)
    #print(stops)
    #exit()
    #print(starts, stops)
    return stabs, starts, stops

def load_jl(jl_file):
    '''
    Load a model from a Julia from into a dictionary
    '''
    with open(jl_file) as f:
        lines = f.readlines()
    model = {'vars':{}, 'pars':{}, 'fns':{}, 'aux':[], 'name':'mmiS'}
    seen_par = 0
    for line in lines:
        if line.lstrip().startswith('du'):
            m = re.search("d(\w+) *= *(.*)", line)
            eq = re.sub("μ", "mu", m.group(2))
            model['vars'][m.group(1)] = eq
        if not seen_par:
            if re.match("p *= *Dict\(", line):
                seen_par = 1
            else:
                continue
        m = re.search("\"(\w+)\" *=> *(.*)$", line)
        par = m.group(1)
        pv = sympy.N(re.sub("\)", "", m.group(2)))
        model['pars'][par] = pv
        if re.search("\)", line):
            seen_par = 0
    for line in lines:
        if re.match(r"p\[\"\w+\"\] *= *", line):
            m = re.search(r"p\[\"(\w+)\"\] *= *(.*)$", line)
            par = m.group(1)
            pv = sympy.N(re.sub("\)", "", m.group(2)))
            model['pars'][par] = pv
    return model

def find_ss_from_bif(data=None, target_pv=None):
    '''
    Find states from bifurcation data based on a target parameter value
    '''
    stabs, starts, stops = find_segs_from_bif(data=data)

    ssdata, ssstabs = [], []
    for start, stop in zip(starts, stops):
        #print(data[start:stop, 1])
        pvs = data[start:stop, 4]
        #print(pvs)
        stab = int(stabs[(start+stop)//2])
        #print(stab)
        pos = np.searchsorted(pvs[::-stab], target_pv)
        phat = pvs[::-stab][pos]
        #print(phat)
        ssdata.append(data[pos+start])
        ssstabs.append(stab)

    ssdata, ssstabs = np.array(ssdata), np.array(ssstabs)
    #print(ssdata, ssstabs)
    return ssdata, ssstabs

def model2te(model, ics={}):
    '''
    Construct Antimony string (for Tellurium) from a model dictionary
    '''
    model_str = '// Reactions\n\t'
    model_str += '\n\n\t' + 'J0: $S -> ' +  list(model['vars'].keys())[0] + '; 0.0\n\t'

    j = 1
    for i, var in enumerate(sorted(model['vars'], reverse=False)):
        if var in model['aux']:
            continue
        de = model['vars'][var]
        model_str += 'J'+ str(j) + ': -> ' + var + '; ' + de + '\n\t'
        j += 1

    model_str += '\n// Aux variables\n\t'

    if 'aux' in model:
        for var in model['aux']:
            model_str += var + ' := ' + model['vars'][var] + '\n\t'

    model_str += '\n// Species Init\n\t'

    for k, v in ics.items():
        model_str += k + ' = ' + str(round(v,4)) + '; '
    model_str += 'S = 0.0'

    model_str += '\n\n// Parameters\n\t'

    for k, v in model['pars'].items():
        model_str += k + ' = ' + str(v) + '; '


    if 'events' in model:
        for ev in model['events']:
            model_str += ev + '; ' + '\n\t'

    #print(model_str)
    #exit()
    r = te.loada(model_str)

    return r

def run_bf(r, auto, dirc="Positive", par="", lims=[0, 1],
        ds=0.001, dsmin=1E-5, dsmax=1, npr=2,
        pre_sim_dur=10, nmx=10000, if_plot=False):
    '''
    Run continuation with a Tellurium model
    '''
    if dirc.lower()[:3] == "pos" or dirc == "+":
        dirc = "Positive"
    elif dirc.lower()[:3] == "neg" or dirc == "-":
        dirc = "Negative"
    # Setup properties
    #auto = Plugin("tel_auto2000")
    auto.setProperty("SBML", r.getCurrentSBML())
    auto.setProperty("ScanDirection", dirc)
    auto.setProperty("PrincipalContinuationParameter", par)
    auto.setProperty("PreSimulation", "True")
    auto.setProperty("PreSimulationDuration", pre_sim_dur)
    auto.setProperty("RL0", lims[0])
    auto.setProperty("RL1", lims[1])
    auto.setProperty("NMX", nmx)
    auto.setProperty("NPR", npr)
    auto.setProperty("KeepTempFiles", True)
    auto.setProperty("DS", ds)
    auto.setProperty("DSMIN", dsmin)
    auto.setProperty("DSMAX", dsmax)
    try:
        auto.execute()
    except Exception as err:
        print(err)
        return [], [], []
    pts     = auto.BifurcationPoints
    if if_plot == True:
        lbl = auto.BifurcationLabels
        biData = auto.BifurcationData
        biData.plotBifurcationDiagram(pts, lbl)
    if not os.path.exists('fort.7'):
        return [], [], []
    else:
        data, bounds, boundsh = extract_data(r)
    #if os.path.exists('fort.7'):
        #os.remove('fort.7')
        #os.remove('fort.8')
    return data, bounds, boundsh

def plot_bfdata58(data, bounds):
    data = np.vstack(data)
    bounds_all = bounds[0]
    for i in range(1, len(bounds)):
        bounds_all = bounds_all + [b+bounds_all[-1] for b in bounds[i]]

    cmap = plt.cm.RdYlGn
    cs = [cmap(0.999), 'gray', cmap(0)]
    fig, (ax0, ax1) = plt.subplots(ncols=2, nrows=1, figsize=(3,1.3), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.15, left=0.23, bottom=0.27, \
            top=0.8, right=0.75, wspace=0.25)

    for ax in (ax0, ax1):
        ax.set_xlim([10, 80])
        ax.set_yscale('symlog', linthreshy=1E-4)

    j = 0
    for i, n in enumerate(bounds_all[1:]):
        st, en = bounds_all[i], n
        a5_arr = data[st:en, 8]
        c8_arr = data[st:en, 18]
        x_arr = data[st:en, 4]
        zorder = 5
        if data[int((st+en)/2), 1] > 0:
            stab = 'u'
            ls = '--'
            c = 'lightblue'
            al = 0.9
            lw = 2
            zorder = 10
        else:
            stab = 's'
            ls = '-'
            lw = 4
            j += 1
            al = 0.9
            if a5_arr[-1] > 0.01 and c8_arr[-1] < 0.07:
                c = cmap(0)
            elif a5_arr[-1] < 0.01 and c8_arr[-1] > 0.07:
                c = cmap(0.999)
            elif a5_arr[-1] < 0.01 and c8_arr[-1] < 0.07:
                c = 'gray'
                lw = 8
                label = None
                zorder = 0
            elif a5_arr[-1] > 0.01 and c8_arr[-1] > 0.07:
                c = cmap(0.5)
                al=1
                lw = 8
                zorder = 0
            else:
                print('Unknown phenotype.')
                c = 'gray'
        print(st, en, stab)
        ax0.plot(data[st:en, 4], data[st:en, 8], ls=ls, c=c, alpha=al, lw=lw, zorder=zorder)
        ax1.plot(data[st:en, 4], data[st:en, 18], ls=ls, c=c, alpha=al, lw=lw, zorder=zorder)
        x_arr[x_arr>80] = np.nan
        x_arr[x_arr<10] = np.nan

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    cax = fig.add_axes([0.8, 0.1, 0.02, 0.35])
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical', cmap=cmap)
    cbar.ax.set_yticklabels([r'$a5^+c8^-$', 'Hybrid', r'$a5^-c8^+$'])
    cax.tick_params(axis='both', which='major', labelsize=8)

    for ax in (ax0, ax1):
        ax.set_xlabel('x/L', size=8)
        ax.xaxis.set_ticks([10, (80-10)/2+10, 80])
        ax.xaxis.set_ticklabels([0, 0.5, 1], size=8)
        ax.tick_params(axis='both', which='major', labelsize=8)

    ax0.set_yticks([1E-1, 1E-3, 0])
    ax0.set_ylabel('Protein\nsteady state (A.U.)', size=8)
    #ax0.set_ylabel('Steady State\nHoxc8 (a.u.)', size=14)
    ax0.set_title('Hoxa5', size=9)
    ax1.set_title('Hoxc8', size=9)

    return fig


class PD_bf:
    def __init__(self, par1_name=None, par2_name=None, model=None, ics=None):
        self.ode = None
        self.ode_te = None
        self.par1_name = par1_name
        #self.par2_name = par2_name
        self.eq_branches = {}
        self.lc_branches = {}
        self.sn_branches = {}
        self.ho_branches = {}
        self.nr_1par_sp = {}
        self.cont_obj = None
        self.dsargs = None

        if model:
            if 1:
                self.ode_te = model2te(model, ics=ics)
            self.model2pd(model, ics=ics, reset_ics=True)

    def model2pd(self, model, ics={}, name=None, reset_ics=False):
        DSargs = args(name=model['name'])
        DSargs.pars = model['pars']
        DSargs.varspecs = model['vars']
        DSargs.ics = ics
        DSargs.fnspecs = model['fns']
        DSargs.algparams = {'refine': True}
        ode = Generator.Radau_ODEsystem(DSargs)
        self.ode = ode
        if reset_ics==True:
            #ode.set(tdomain=[0, 100], tdata=[0, 100])
            #trj = testDS.compute('prerun').sample(dt=0.2)
            #self.ode.set(ics=trj[-1])
            self.reset_ics(ics)
        cont = ContClass(ode)
        self.cont_obj = cont
        self.dsagrs = DSargs

    def set(self,pars={}):
        self.ode.set(pars=pars)
        self.cont_obj.model.set(pars=pars)
        for k, v in pars.items():
            self.ode_te[k] = v
        self.reset_ics()

    def reset_ics(self, ics=None):
        if not ics:
            ics = self.ode.query(querykey='ics')
        #print('aaaa')
        #self.ode.set(tdomain=[0, 10000], tdata=[0, 10000], ics=ics)
        #trj = self.ode.compute('prerun').sample(dt=0.2)
        #self.ode.set(ics=trj[-1])
        if 1:
            m = self.ode_te.simulate(0, 20000, 100)
            ics_new = dict(zip(self.ode_te.fs(), m[-1][1:]))
            self.ode.set(ics=ics_new)
        if self.cont_obj:
            #self.cont_obj.model.set(ics=trj[-1])
            self.cont_obj.model.set(ics=ics_new)

    def simulate(self):
        trj = self.ode.compute('run').sample(dt=0.2)
        return trj

    def add_1par_branch(self, ics=None, par1=None,
            ss=3e-4, maxss=3e-3, minss=3e-5, np=1000, dirc='f',\
            #ss=1e-2, maxss=0.1, minss=1e-3, np=800, dirc='f',\
            bp_pt=None, **kwargs):
        ics = self.ode.query(querykey='ics')
        self.reset_ics()
        self.ode.set(tdomain=[0, 100], tdata=[0, 100], ics=ics)
        branch_num = len(self.eq_branches.keys()) + 1
        branch_name = 'EQ' + str(branch_num)
        self.eq_branches[branch_name] = 1
        print('Computing branch ' + branch_name)
        PCargs = args(name=branch_name, type='EP-C')
        PCargs.freepars = [par1]
        self.par1_name = par1
        #PCargs.initpoint = PyDSTool.Point({'coorddict': new_ics})
        PCargs.StepSize = ss
        PCargs.MaxStepSize = maxss
        PCargs.MinStepSize = minss
        PCargs.MaxNumPoints = np
        PCargs.LocBifPoints = 'all'
        PCargs.verbosity = 2
        PCargs.SaveEigen = True
        PCargs.SaveJacobian = True
        self.cont_obj.newCurve(PCargs)

        print('Computing curve...')
        start = perf_counter()
        if dirc == 'f':
            self.cont_obj[branch_name].forward()
        elif dirc == 'b':
            self.cont_obj[branch_name].backward()
        else:
            self.cont_obj[branch_name].forward()
            self.cont_obj[branch_name].backward()
            print('Unspecified direction\n')
        print('done in %.3f seconds!' % (perf_counter()-start))

        lcpt, stpt = 'H2', 'H1'
        if self.cont_obj[branch_name].getSpecialPoint(lcpt):
            print('HP point found')
            # Limit cycle curve. Turn it off if only HB and LP points are needed.
            lcbranch_num = len(self.lc_branches.keys()) + 1
            lcbranch_name = 'LC' + str(lcbranch_num)
            self.lc_branches[branch_name] = 1
            print('Computing branch ' + lcbranch_name)
            PCargs.name = lcbranch_name
            PCargs.type = 'LC-C'
            #PCargs.MaxNumPoints = 600 #// 10
            PCargs.MaxNumPoints = 1600 #// 10
            PCargs.NumIntervals = 60
            PCargs.NumCollocation = 7
            PCargs.initpoint = branch_name + ':' + lcpt
            PCargs.SolutionMeasures = 'all'
            PCargs.NumSPOut = 200
            PCargs.FuncTol = 1e-4
            PCargs.VarTol = 1e-4
            PCargs.TestTol = 1e-3
            PCargs.SaveEigen = True
            #PCargs.StopAtPoints = [stpt]
            self.cont_obj.newCurve(PCargs)

            print('Computing limit-cycle curve...')
            start = perf_counter()
            self.cont_obj[lcbranch_name].forward()
            #self.cont_obj['LC1'].backward()
            print('done in %.3f seconds!' % (perf_counter()-start))

            #self.cont_obj[lcbranch_name].display(('mu','R_min'), stability=True, figure=1) # Get the lower branch of the cycle.
            #self.cont_obj[lcbranch_name].display(('mu','R'), stability=True, figure=1)

            #self.cont_obj[lcbranch_name].display(stability=True, figure='new', axes=(1,2,1))
            #self.cont_obj[lcbranch_name].plot_cycles(coords=('R','r'), linewidth=1, axes=(1,2,2), figure='fig2')

        return PCargs

    def get_curve(self, branch_name, var=None):
        vars_all = self.ode.query(querykey='vars')
        if not 'lc' in branch_name:
            if branch_name == 'last':
                branch_name = 'EQ' + str(len(self.eq_branches.keys()))
            print(type(self.cont_obj[branch_name].sol))
            if not isinstance(self.cont_obj[branch_name].sol, PyDSTool.Points.Pointset) or not 'EP' in self.cont_obj[branch_name].sol[0].labels:
                # TODO: loop over sol to check if 'EP' is in labels
                curves_vars = {}
                for v in vars_all:
                    curves_vars[v] = [0]
                curves_vars['if_imag'] = [0]
                curves_vars['stab'] = [0]
                curves_vars['par'] = [0]
                return pd.DataFrame(curves_vars)
            stop_id = len(self.cont_obj[branch_name].sol[self.par1_name])
            #curve_evals = np.array([x.labels['EP']['data'].evals for x in self.cont_obj[branch_name].sol])
            curve_evals = []
            for x in self.cont_obj[branch_name].sol:
                if 'EP' not in x.labels:
                    break
                else:
                    curve_evals.append(x.labels['EP']['data'].evals)
            curve_evals = np.array(curve_evals)
            #curve_stabs = np.array([x.labels['EP']['stab'] for x in self.cont_obj[branch_name].sol])
            curve_stabs = []
            for i, x in enumerate(self.cont_obj[branch_name].sol):
                if 'EP' not in x.labels:
                    stop_id = i
                    print('Stop at ' + str(i))
                    break
                else:
                    curve_stabs.append(x.labels['EP']['stab'])
            curve_stabs = np.array(curve_stabs)
            if_imag = (np.imag(curve_evals) != 0).sum(axis=1)
            #if_imag = (np.imag(curve_evals) >0.001).sum(axis=1)
            for i, im, in enumerate(if_imag):
                if i == 0 or i == len(if_imag) - 1:
                    continue
                else:
                    if if_imag[i-1] == 0 and if_imag[i+1] == 0:
                        if_imag[i] = 0
            curves_vars = {}
            for v in vars_all:
                curves_vars[v] = self.cont_obj[branch_name].sol[v][:stop_id]
            curves_vars['if_imag'] = if_imag
            curves_vars['stab'] = curve_stabs
            curves_vars['par'] = self.cont_obj[branch_name].sol[self.par1_name][:stop_id]
            curvedf = pd.DataFrame(curves_vars)
        elif branch_name == 'lclast':
            branch_name = 'LC' + str(len(self.lc_branches.keys()))
            curve_stabs = np.array([x.labels['LC']['stab'] for x in self.cont_obj[branch_name].sol])
            curves_vars = {}
            for k in self.cont_obj[branch_name].sol.keys():
                curves_vars[k] = self.cont_obj[branch_name].sol[k]
            curves_vars['stab'] = curve_stabs
            curves_vars['par'] = self.cont_obj[branch_name].sol[self.par1_name]
            curvedf = pd.DataFrame(curves_vars)
        return curvedf

        #fig, ax = plt.subplots(figsize=(5, 3))
        #ax.plot(curve_mu, np.ma.masked_where(np.logical_not(if_imag), curve_R), c='gold', lw=5, zorder=-1)
        #ax.plot(curve_mu, np.ma.masked_where(curve_stabs!='S', curve_R), c='k', lw=2, zorder=0)
        #ax.plot(curve_mu, np.ma.masked_where(curve_stabs=='S', curve_R), c='k', ls='--', lw=2, zorder=0)
        #ax.plot(curve_mu, curve_R, c='k', lw=1, zorder=0)
        #plt.show()
