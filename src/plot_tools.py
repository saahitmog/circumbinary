import os
import numpy as np
import pandas as pd
from rebound import Simulation
from matplotlib import pyplot as plt
from matplotlib import patches as pth
from multiprocessing.pool import Pool
from scipy.integrate import solve_ivp
from src.functions import p2a, a2p, R_sun, HW99

color_A = '#005AB5'
color_B = '#DC3220'
color_C = '#FFB000'

def init(M_A: float, M_B: float, a_b: float, a_p: float, 
         inc_b: float, inc_p: float, Omega_b: float = 0,
         Omega_p: float = 0, e_b: float = 0, e_p: float = 0,
         omega_p: float = 0, omega_b: float = 0) -> Simulation:

    sim = Simulation()
    sim.add(m=M_A)
    sim.add(m=M_B, a=a_b, inc=inc_b, 
            Omega=Omega_b, e=e_b, omega=omega_b)
    sim.add(a=a_p, inc=inc_p, Omega=Omega_p, e=e_p, omega=omega_p)
    sim.move_to_com()
    # sim.ri_ias15.min_dt = 1e-3
    
    return sim

def integrate(sim: Simulation, P: float = None, 
              N: float = None) -> tuple[np.array]:
    if not P:
        P = sim.particles[2].P
    if N:
        P *= N
    x, y, z, dt = ([[] for _ in range(sim.N)], 
                   [[] for _ in range(sim.N)], 
                   [[] for _ in range(sim.N)], 
                   [])
    while sim.t < P:
        sim.step()
        for i, p in enumerate(sim.particles):
            x[i].append(p.x)
            y[i].append(p.y)
            z[i].append(p.z)
            dt.append(sim.dt)
    x, y, z = np.array(x), np.array(y), np.array(z)
    sim = None
    return x, y, z

def plot_system(M_A: float, M_B: float, a_b: float, r_a: float, 
                inc_b: float, Omega_p: float = 0, Omega_b: float = 0, 
                e_b: float = 0, omega_p: float = 0, omega_b: float = 0, 
                N: float = None, dinc: float = 0, fancy: bool = False, 
                xlim: float = 0.5, ylim: float = 0.5, offset: int = None, 
                title: str = None, scaled: bool = True) ->  None:
  
    sim = init(M_A=M_A, M_B=M_B, a_b=a_b, a_p=r_a*a_b, e_b=e_b, 
               inc_b=np.deg2rad(inc_b), inc_p=np.deg2rad(inc_b+dinc), 
               Omega_b=np.deg2rad(Omega_b), Omega_p=np.deg2rad(Omega_p), 
               omega_p=np.deg2rad(omega_p), omega_b=np.deg2rad(omega_b))
    x, y, z = integrate(sim, N=N)
    
    if not fancy:
        fig = plt.figure()
    else: fig = plt.figure(figsize=(7, 3))
    ax = fig.add_subplot()
    ax.plot(x[0], z[0], c='r', lw=1, alpha=0.7)
    ax.plot(x[1], z[1], c='b', lw=1, alpha=0.7)
    ax.plot(x[2], z[2], c='k', lw=1, alpha=0.7)
    
    if fancy:
        try:    
            center = abs(x[2]) - min(x[2], key=abs)
            cond = (center<0.02) & (z[2]<=0)
            p_z = z[2, cond][0]
            planet = (0, p_z)
            planet = plt.Circle(planet, 0.01, color='black', label='Planet')
            ax.add_artist( planet )
        except:
            print('no planet')

        idx = np.extract(abs(x[0]) == abs(x[0]).min(), 
                         np.arange(x[0].size)) + offset

        _x = x[0][idx]
        _z = z[0][idx]
        A = (_x, _z)
        A = plt.Circle(A, 0.03, color='red',
                       alpha=0.25, label='Star A')

        _x = x[1][idx]
        _z = z[1][idx]
        B = (_x, _z)
        B = plt.Circle(B, radius=0.03, color='blue', 
                       alpha=0.25, label='Star B')
        
        ax.add_artist( A )
        ax.add_artist( B )
        
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-ylim, ylim)
        ax.legend(loc='upper left', fontsize='small')

    ax.grid(True)
    if scaled: ax.axis('scaled')
    if title is None: title = "View of the System along the y-axis"
    ax.set_title(title)
    ax.set_xlabel("x [AU]", fontsize='large')
    ax.set_ylabel("z [AU]", fontsize='large')
    plt.show()

def fOmega(e):
    return (1+3*e**2+3/8*e**4)/(1-e**2)**(9/2)

def fOmega_e(e):
    return (1+3/2*e**2+1/8*e**4)/(1-e**2)**5

def fN(e):
    return (1+15/2*e**2+45/8*e**4+5/16*e**6)/(1-e**2)**6

def fN_e(e):
    return (1+15/4*e**2+15/8*e**4+5/64*e**6)/(1-e**2)**(13/2)

def fN_a(e):
    return (1+31/2*e**2+255/8*e**4+185/16*e**6+25/64*e**8)/(1-e**2)**(15/2)

def F_e(e):
    return (fOmega_e(e)*fN(e)/fOmega(e))-18*fN_e(e)/11

def F_a(e):
    return 4/11*(fN(e)**2/fOmega(e)-fN_a(e))

def tidal_circ(t, orb, mu, eta, Penv=4):
	e, P = orb
	tc = 0.3*(P/Penv)**eta
	dedt = e*mu*(1.+mu)/tc*F_e(e)
	dPdt = 1.5*P*mu*(1.+mu)/tc*F_a(e)
	return [dedt, dPdt]

def solve_circularization(inp, n_eval=1, t_age=None):
    e, P, mu, eta, Penv = inp
    if t_age is None: t_age = np.random.default_rng().uniform(1, 10)
    if n_eval == 1: t_eval = [t_age]
    else: t_eval = np.linspace(0, t_age, n_eval)
    sol = solve_ivp(tidal_circ, t_span=(0, t_age), 
                    y0=(e, P), t_eval=t_eval, 
                    args=(mu, eta, Penv), method='LSODA')
    if n_eval == 1: return sol.y.flatten()
    else: return sol.y

def roche_limit(mu, e, P):
    return 0.44*mu**-0.33/(1+1/mu)**0.2 * p2a(P, 1)*(1-e)

def binary_pop(eta, N: int = 1_000):
    e = np.random.default_rng().beta(1.75, 2.01, N)
    P_min, P_max = 1, 200
    l = np.random.default_rng().triangular(0, 1, 1, N)
    P = P_max**l*P_min**(1-l)
    mu = np.random.default_rng().triangular(0, 1, 1, N)

    roche_mask_A = R_sun >= roche_limit(1/mu, e, P)
    roche_mask_B = R_sun*mu**0.8 >= roche_limit(mu, e, P)
    s = ~(roche_mask_A | roche_mask_B)

    eta = np.full_like(e, eta)
    inp = np.array((e, P, mu, eta)).T

    with Pool() as pool:
        ec, Pc = np.array(pool.map(solve_circularization, inp)).T
    return ec, Pc, e, P, mu, s

def read_df(fname, key: str='data'):
    with pd.HDFStore(fname) as hdf:
        return hdf[key]

def system_pop(n_pop: int, eta: float=4.5, Penv: float=4, Pmin: float=3):
    rng = np.random.default_rng()
    e = rng.beta(1.75, 2.01, n_pop)

    P_min, P_max = Pmin, 200
    l = rng.triangular(0, 1, 1, n_pop)
    P = P_max**l*P_min**(1-l)

    mu = rng.triangular(0, 1, 1, n_pop)
    eta = np.full_like(e, eta)
    Penv = np.full_like(e, Penv)
    inp = np.array((e, P, mu, eta, Penv)).T

    with Pool() as pool:
        e_circ, P_circ = np.array(pool.map(solve_circularization, inp)).T

    UPPER_ANGLE = 30
    arg = np.sin(np.deg2rad(UPPER_ANGLE))
    i = np.arcsin(rng.uniform(0, arg, n_pop)) + np.pi/2
    omega = rng.uniform(0, 2*np.pi, n_pop)
    Omega = rng.uniform(0, 2*np.pi, n_pop)

    r_a = R_sun
    r_b = R_sun*mu**0.8
    a = p2a(P_circ, 1)

    corr = (1-np.abs(e_circ*np.cos(omega)))/(1-e_circ**2)
    eclipsing = abs(np.sin(i-np.pi/2)) <= (r_a+r_b)/a * corr

    roche_limit = 0.44*mu**-0.33/(1+1/mu)**0.2
    pericenter = a*(1-e_circ)
    stable = r_a < roche_limit*pericenter

    data = {'eccentricity': e, 'period': P, 'mass_ratio': mu,
            'circ_eccentricity': e_circ, 'circ_period': P_circ,
            'inclination': np.rad2deg(i), 'omega': omega, 'Omega': Omega,
            'radius_a': r_a, 'radius_b': r_b, 'semimajor_axis': a,
            'eclipsing?': eclipsing, 'stable?': stable}
    
    return pd.DataFrame(data=data)

def continuous_segments(arr, segment_mask):
    segments = []
    
    start_indices = np.where(segment_mask & ~np.roll(segment_mask, 1))[0]
    end_indices = np.where(~segment_mask & np.roll(segment_mask, 1))[0]
    
    if segment_mask[0]:
        start_indices = np.concatenate(([0], start_indices))
    if segment_mask[-1]:
        end_indices = np.concatenate((end_indices, [len(arr)]))
    
    for start_idx, end_idx in zip(start_indices, end_indices):
        segments.append(arr[start_idx:end_idx])
    
    return segments

def display(files, nbins=50):
    if not isinstance(files, list): files = [files]
    n = len(files)
    fig, ax = plt.subplots(2, 5, sharex=True, sharey=True, 
                           layout='constrained', figsize=(40,7))
    ax = ax.flatten()

    for file, axis in zip(files, ax):
        file = './data/' + file
        with pd.HDFStore(file) as hdf:
            key = 'data'
            df = hdf[key]
            metadata = hdf.get_storer(key).attrs.metadata

        df = df.fillna(value=np.nan)
        R = df.Radius_A + df.Radius_B
        a = df.Binary_Semimajor_Axis
        e = df.Binary_Eccentricity
        i = np.rad2deg(df.Binary_Inclination)-90
        p = df.Binary_Period
        pi = df.Initial_Binary_Period
        pf = df.Final_Binary_Period
        omega = np.deg2rad(df.Binary_omega)
        ecosw = e * np.cos(omega)
        corr = (1-np.abs(ecosw))/(1-e**2)

        eclipsing = abs(np.sin(np.deg2rad(i))) <= R/a * corr
        stable = df.Transit_Value >= 0
        transiting = df.Transit_Value == 2

        plt1 = p[~eclipsing & transiting]
        plt2 = p[eclipsing & stable]

        logbins = np.logspace(np.log10(np.amin(np.concatenate((plt1,plt2,)))), 
                              np.log10(np.amax(np.concatenate((plt1,plt2,)))), nbins)
        bins = logbins
        axis.hist(plt1, bins, label=f'Transiting ({plt1.size})', 
                  histtype='step', color='r', lw=1.5, density=True)
        axis.hist(plt2, bins, label=f'Eclipsing ({plt2.size})',
                  histtype='step', color='k', lw=1.5, density=True)
        axis.axvline(6, ls='--')
        axis.set_title(r'$N$ = ' + str(metadata['N'])
                       + r'$\ \vert\ \eta$ = ' + str(metadata['eta'])
                       + r'$\ \vert\ \alpha$ = ' + str(metadata['alpha'])
                       + r'$\ \vert\ \sigma_{\Delta i}$ = '+str(metadata['sigma_di'])
                       , fontsize=22)
        axis.tick_params(labelsize=15)
        if bins is logbins:
            axis.set_xscale('log')
        axis.legend(loc=1, fontsize=15)
    
    fig.suptitle('Distribution of Eclipsing Binary Periods', fontsize=30)
    fig.supylabel('Number of Detected Planets', fontsize=22)
    fig.supxlabel('$P_b$ [days]', fontsize=22)
    plt.show()

def extract_eclipsing_transiting(df):
    df = df[df.Transit_Value == 2]
    R = df.Radius_A + df.Radius_B
    a = df.Binary_Semimajor_Axis
    e = df.Binary_Eccentricity
    omega = np.deg2rad(df.Binary_omega.astype(float))
    ecosw = e*np.cos(omega)
    corr = (1-np.abs(ecosw))/(1-e**2)
    eclipsing = (df.Binary_Inclination-np.pi/2) <= R/a * corr
    df = df[eclipsing]
    return df

def ecosw_err(e, w, e_err, w_err):
    x1 = e_err * np.cos(np.deg2rad(w))
    x2 = np.deg2rad(w_err) * e*np.sin(np.deg2rad(w))
    return np.hypot(x1, x2)

def add_arrow(line, position=None, direction='right', size=14, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()

    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    try:
        line.axes.annotate('',
            xytext=(xdata[start_ind], ydata[start_ind]),
            xy=(xdata[end_ind], ydata[end_ind]),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=0),
            size=size
        )
    except:
        pass

windemuth = pd.read_csv('data/windemuth.posteriors', sep=' ', header=1)

files_control = ('data/e45_d10_a05_m3_0.h5', 
                 'data/e45_d10_a05_m1_0.h5',
                 'data/e45_d10_a05_m0_0.h5')
hist_titles_control = ('Control Case',
                       'Planets Form\nAfter Decay', 
                       'Planets Form\nBefore Decay')

files_eta = ('data/e20_d10_a05_m0_0.h5', 
             'data/e45_d10_a05_m0_0.h5',
             'data/e70_d10_a05_m0_0.h5')
hist_titles_eta = (r'$\eta=2.0$', r'$\eta=4.5$', r'$\eta=7.0$')

files_alpha = ('data/e45_d10_a10_m0_0.h5', 
               'data/e45_d10_a05_m0_0.h5',
               'data/e45_d10_a00_m0_0.h5',
               'data/e45_d10_a-05_m0_0.h5')
hist_titles_alpha = (r'$\alpha=1.0$', r'$\alpha=0.5$', 
                     r'$\alpha=0.0$', r'$\alpha=-0.5$')

files_sigma = ('data/e45_d03_a05_m0_0.h5', 
              'data/e45_d10_a05_m0_0.h5',
              'data/e45_d30_a05_m0_0.h5')
hist_titles_sigma = (r'$\sigma_m=0.3$', r'$\sigma_m=1.0$', 
                     r'$\sigma_m=3.0$')

files_penv = ('data/e45_d10_a05_m0_0.h5',
              'data/e45_d10_a05_p50_m0_0.h5',
              'data/e45_d10_a05_p60_m0_0.h5',
              # 'data/e45_d10_a05_p80_m0_0.h5',
              # 'data/e45_d10_a05_p120_m0_0.h5'
              )
hist_titles_penv = (r'$(P_{\rm env}, P_{\rm min})=(4.0, 3.0)$ days',
                    r'$(P_{\rm env}, P_{\rm min})=(5.0, 3.75)$ days',
                    r'$(P_{\rm env}, P_{\rm min})=(6.0, 4.5)$ days',
                    # r'$(P_{\rm env}, P_{\rm min})=(8.0, 6.0)$ days', 
                    # r'$(P_{\rm env}, P_{\rm min})=(12.0, 9.0)$ days'
                    )

all_dfs = [[df for df in [read_df(file) for file in files]] 
            for files in [files_eta, files_sigma, files_alpha]]
all_hist_titles = (hist_titles_eta, hist_titles_sigma, hist_titles_alpha)

all_mode0_df = pd.concat(read_df(f'data/{f}') for f in os.listdir('data') if f[-6] == '0')

observations = {'Kepler 16b' : {'M_A': 0.69,"M_B": 0.2,'a_b': 0.22,
                                'a_p': 0.7,'e_b': 0.16, 'outlier': 0, 
                                'ecosw': 0.0181},
                'Kepler 34b' : {'M_A': 1.05,"M_B": 1.02,'a_b': 0.23,
                                'a_p': 1.09,'e_b': 0.52, 'outlier': 0,
                                'ecosw': 0.1658},
                'Kepler 35b' : {'M_A': 0.89,"M_B": 0.81,'a_b': 0.18,
                                'a_p': 0.6,'e_b': 0.14, 'outlier': 0,
                                'ecosw': 0.0086},
                'Kepler 38b' : {'M_A': 0.95,"M_B": 0.25,'a_b': 0.15,
                                'a_p': 0.46,'e_b': 0.1, 'outlier': 0,
                                'ecosw': abs(0.1032*np.cos(np.deg2rad(268.68)))},
                'Kepler 47b' : {'M_A': 1.04,"M_B": 0.36,'a_b': 0.084,
                                'a_p': 0.3,'e_b': 0.023, 'outlier': 0,
                                'ecosw': 0.0199},
                'Kepler 47c' : {'M_A': 1.04,"M_B": 0.36,'a_b': 0.084,
                                'a_p': 0.99,'e_b': 0.023, 'outlier': 1,
                                'ecosw': 0.0199}, 
                'Kepler 47d' : {'M_A': 1.04,"M_B": 0.36,'a_b': 0.084,
                                'a_p': 	0.699,'e_b': 0.023, 'outlier': 1,
                                'ecosw': 0.0199},
                'Kepler 64b' : {'M_A': 1.53,"M_B": 0.41,'a_b': 0.17,
                                'a_p': 0.63,'e_b': 0.21, 'outlier': 0,
                                'ecosw': abs(0.2117*np.cos(np.deg2rad(217.6)))},
                'Kepler 413b' : {'M_A': 0.82,"M_B": 0.54,'a_b': 0.1,
                                 'a_p': 0.36,'e_b': 0.037, 'outlier': 0,
                                 'ecosw': 0.0062},
                'Kepler 453b' : {'M_A': 0.93,"M_B": 0.19,'a_b': 0.18,
                                 'a_p': 0.79,'e_b': 0.051, 'outlier': 0,
                                 'ecosw': 0.0063},
                'Kepler 1647b' : {'M_A': 1.22,"M_B": 0.97,'a_b': 0.13,
                                  'a_p': 2.72,'e_b': 0.16, 'outlier': 2,
                                  'ecosw': 0.1386},
                'Kepler 1661b' : {'M_A': 0.84,"M_B": 0.26,'a_b': 0.187,
                                  'a_p': 0.633,'e_b': 0.112, 'outlier': 0,
                                  'ecosw': abs(0.112*np.cos(np.deg2rad(36.4)))},
                'TOI 1338' : {'M_A': 1.127,"M_B": 0.3128,'a_b': 0.1321,
                              'a_p': 0.4491,'e_b': 0.156, 'outlier': 0,
                              'ecosw': abs(0.1560*np.cos(np.deg2rad(117.568)))}
                }

observations_df = pd.DataFrame({'name': observations.keys()})
for key in observations['Kepler 16b']:
    observations_df[key] = [observations[i][key] for i in observations]

for name, info in observations.items():
    info['P_b'] = a2p(info['a_b'], info['M_A']+info['M_B'])
    info['P_p'] = a2p(info['a_p'], info['M_A']+info['M_B'])

obs_P_b = [info['P_b'] for _, info in observations.items()]
obs_P_p = [info['P_p'] for _, info in observations.items()]

all_systems = observations_df.copy()
p_ratio_obs = []
for i, planet in all_systems.iterrows():
    mu = planet['M_B']/(planet['M_B']+planet['M_B'])
    a_HW = HW99(mu, planet['e_b'], planet['a_b'])
    P_HW = a2p(a_HW, planet['M_A']+planet['M_B'])
    P_p = a2p(planet['a_p'], planet['M_A']+planet['M_B'])
    p_ratio_obs = np.append(p_ratio_obs, P_p/P_HW)
p_ratio_obs_all = np.sort(p_ratio_obs)

no_outlier_systems = observations_df[observations_df['outlier'] == 0]
p_ratio_obs = []
for i, planet in no_outlier_systems.iterrows():
    mu = planet['M_B']/(planet['M_B']+planet['M_B'])
    a_HW = HW99(mu, planet['e_b'], planet['a_b'])
    P_HW = a2p(a_HW, planet['M_A']+planet['M_B'])
    P_p = a2p(planet['a_p'], planet['M_A']+planet['M_B'])
    p_ratio_obs = np.append(p_ratio_obs, P_p/P_HW)
p_ratio_obs_no_outliers = np.sort(p_ratio_obs)