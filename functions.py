import numpy as np
import shapely as sh
from rebound import Simulation
from scipy.integrate import solve_ivp
from astropy import constants as const
from astropy import units as u

R_sun = u.R_sun.to('AU')
yr = u.yr.to('day')

def HW99(mu: float, e: float, a_b: float) -> float:
    return (1.6+4.12*mu+5.1*e-4.27*mu*e
            -2.22*e**2-5.09*mu**2+4.61*e**2*mu**2)*a_b

def p2a(P, M):
    ''' P (days), M (M_sun) ''' 
    a3 = const.GM_sun*M*(P*u.day/(2*np.pi))**2
    return np.cbrt(a3).to('AU').value

def a2p(a, M):
    ''' a (AU), M (M_sun) '''
    P = 2*np.pi*np.sqrt((a*u.AU)**3/(const.GM_sun*M))
    return P.to('day').value

def rotate(theta, axis):
    matrix = np.empty((3,3))
    if axis == 'x':
        matrix = np.array([[1,             0,              0],
                           [0, np.cos(theta), -np.sin(theta)],
                           [0, np.sin(theta),  np.cos(theta)]])
    elif axis =='y':
        matrix = np.array([[ np.cos(theta), 0, np.sin(theta)],
                           [             0, 1,             0],
                           [-np.sin(theta), 0, np.cos(theta)]])
    elif axis == 'z':
        matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                           [np.sin(theta),  np.cos(theta), 0],
                           [            0,              0, 1]])
    else:
        raise ValueError('Invalid Axis')
    return matrix

def sky_transform(i_b, Omega_b, omega_b, di, Omega_p, omega_p):
    A = np.matmul(rotate(-i_b, 'x'), rotate(-Omega_b, 'z'))
    x, y, z = np.matmul(rotate(-omega_b, 'z'), A)

    A = np.matmul(rotate(di, 'x'), rotate(omega_p, 'z'))
    u, v, l = np.matmul(rotate(Omega_p, 'z'), A)

    i_p_sky = np.arccos(np.dot(z, l))

    n = np.cross(z, l)
    n = n/np.linalg.norm(n)

    Omega_p_sky = np.arccos(np.dot(x, n))
    if np.dot(y, n) < 0:
        Omega_p_sky = 2*np.pi - Omega_p_sky

    omega_p_sky = np.arccos(np.dot(n, u))
    if np.dot(u, n) < 0:
        omega_p_sky = 2*np.pi - omega_p_sky

    return i_p_sky, Omega_p_sky, omega_p_sky

def consecutive_masked_runs(arr, mask):
    transitions = np.where(np.diff(np.concatenate(([0], mask, [0]))) != 0)[0]
    masked_runs = [arr[transitions[i]:transitions[i + 1]] 
                    for i in range(0, len(transitions), 2) 
                    if mask[transitions[i]]]
    return masked_runs

def tidal_circ(t, orb, mu, eta):
    def fN(e):
        return (1+15/2*e**2+45/8*e**4+5/16*e**6)/(1-e**2)**6
    def fOmega(e):
        return (1+3*e**2+3/8*e**4)/(1-e**2)**(9/2)
    def F_e(e):
        def fN_e(e):
            return (1+15/4*e**2+15/8*e**4+5/64*e**6)/(1-e**2)**(13/2)
        def fOmega_e(e):
            return (1+3/2*e**2+1/8*e**4)/(1-e**2)**5
        return (fOmega_e(e)*fN(e)/fOmega(e))-18*fN_e(e)/11
    def F_a(e):
        def fN_a(e):
            return ((1+31/2*e**2+255/8*e**4+185/16*e**6+25/64*e**8)
                    / (1-e**2)**(15/2))
        return 4/11*(fN(e)**2/fOmega(e)-fN_a(e))   

    e, P = orb
    tc = 0.3*(P/4.)**eta
    dedt = e*mu*(1.+mu)/tc*F_e(e)
    dPdt = 1.5*P*mu*(1.+mu)/tc*F_a(e)
    return [dedt, dPdt]

def init(M_A: float, M_B: float, a_bin: float, a_p: float, 
            inc_bin: float, inc_p: float, Omega_bin: float = 0,
            Omega_p: float = 0, e_B: float = 0, e_p: float = 0,
            omega_b: float = 0, omega_p: float = 0) -> Simulation:
    """Returns a Simulation object of all 3 objects in the system.

    Parameter
    ---------
    M_A : float
        Mass of star A (solar masses)
    M_B : float
        Mass of star B (solar masses)
    a_bin : float
        Semi-major axis of binary (AU)
    a_p : float
        Semi-major axis of planet (AU)
    inc_bin : float
        Inclination of binary (rad)
    Omega_bin : float
        Longitude of ascending node of binary (rad)
    inc_p : float
        Inclination of planet (rad)
    Omega_p : float
        Longitude of ascending node of planet (rad)
    e_B : float (default 0.0)
        Orbital eccentricity of binary
    e_p : float (default 0.0)
        Orbital eccentricity of planet
    omega : float (default 0.0)
        Argument of pericenter (rad)
    """
    sim = Simulation()
    sim.units = ('AU','yr','Msun')
    sim.add(m=M_A)
    sim.add(m=M_B, a=a_bin, inc=inc_bin, Omega=Omega_bin, e=e_B, omega=omega_b)
    sim.add(a=a_p, inc=inc_p, Omega=Omega_p, e=e_p, omega=omega_p)
    sim.move_to_com()
    sim.ri_ias15.min_dt = 1e-8
    return sim

def integrate(sim: Simulation, P: float) -> tuple[np.array]:
    """Returns arrays of the x and y coordinates of the particles 
    in the Simulation object after being integrated over 'Norbits' 
    planet orbits.

    Parameter
    ---------
    sim : Simulation
        Simulation object
    """
    positions = [[] for _ in range(sim.N)]

    while sim.t < P:
        sim.step()
        for i, particle in enumerate(sim.particles):
            positions[i].append([particle.x, particle.y, particle.z])

    positions = np.array(positions)
    sim = None

    return tuple(positions.transpose(2, 0, 1))

def check_transit(R_A: float, R_B: float, R_p: float, 
                    x: np.array, y: np.array, z: np.array) -> int:
    """Returns an int corresponding the number of stars 
    in the binary that the planet transits.

    Parameter
    ---------
    R_A : float
        Radius of star A (AU)
    R_B : float
        Radius of star B (AU)
    R_p : float
        Radius of planet (AU)
    x : np.array
        x-coordinates of the bodies in the system
    y : np.array
        y-coordinates of the bodies in the system
    """

    x_p, y_p, z_p = x[2], y[2], z[2]

    rngA = sh.LinearRing(np.array((x[0], y[0])).T)
    rngB = sh.LinearRing(np.array((x[1], y[1])).T)

    Rb = max(R_A, R_B) + R_p
    bounds = np.vstack((np.array(rngA.bounds).reshape(2,2), 
                        np.array(rngB.bounds).reshape(2,2)))
    lb, ub = bounds.min(0)-Rb, bounds.max(0)+Rb

    mask = (x_p>=lb[0]) & (x_p<=ub[0]) & (y_p>=lb[1]) & (y_p<=ub[1]) & (z_p>0)
    x_arcs_masked = consecutive_masked_runs(x_p, mask)
    y_arcs_masked = consecutive_masked_runs(y_p, mask)

    results = []
    for _x, _y in zip(x_arcs_masked, y_arcs_masked):
        if _x.size == 1 and _y.size == 1: 
            arc = sh.Point(_x, _y)
        else:
            arc = sh.LineString(np.array((_x, _y)).T)
        transit_A = sh.dwithin(rngA, arc, R_A)
        transit_B = sh.dwithin(rngB, arc, R_B)
        result = int(transit_A) + int(transit_B)
        results.append(result)

    return max(results, default=0)

def sample_check(eta, sigma_di, alpha, mode):

    rng = np.random.default_rng()

    # Sample Binary Parameters
    q = rng.triangular(0, 1, 1)
    M_A, M_B, M_p = 1, q, 0
    R_A, R_B, R_p = R_sun, R_sun*q**0.8, 0

    P_min, P_max = 3, 200
    l = rng.triangular(0, 1, 1)
    P_b = P_max**l*P_min**(1-l)
    e_b = rng.beta(1.75, 2.01)
    a_b = p2a(P_b, M_A)

    # Tidally Circularize
    t_age = rng.uniform(1, 10)
    sol = solve_ivp(tidal_circ, (0, t_age), (e_b, P_b), t_eval=[t_age], 
                    args=(q, eta), method='LSODA')
    e_b_circ, P_b_circ = sol.y.flatten()
    e_b_circ = abs(e_b_circ)

    a_b_circ = p2a(P_b_circ, M_A)

    roche_limit = 0.44*q**-0.33/(1+1/q)**0.2
    pericenter = a_b_circ*(1-e_b_circ)

    if R_A >= roche_limit*pericenter: # binary roche limit          
        system = np.array([P_b, P_b_circ, None, a_b_circ, 
                            None, None, None, M_A, M_B, 
                            R_A, R_B, e_b, e_b_circ, None, 
                            None, None, None, None, None, None, 
                            None, None, None, None, None, -2])
        return system

    mu = q/(1+q)
    a_HW = HW99(mu, e_b, a_b)
    P_HW = a2p(a_HW, M_A+M_B)
    if mode == 1: # planets populate down to circularized HW limit
        a_HW = HW99(mu, e_b_circ, a_b_circ)
        P_HW = a2p(a_HW, M_A+M_B)
    if mode == 2: # no circularization
        a_b_circ, e_b_circ, P_b_circ = a_b, e_b, P_b

    # Sample Planet Parameters
    P_min, P_max = P_HW, 100*P_HW
    u = rng.uniform()
    if alpha:
        P_p = ((P_max**alpha-P_min**alpha)*u + P_min**alpha)**(1/alpha)
    else:
        P_p = P_min*(P_max/P_min)**u
    a_p = p2a(P_p, M_A+M_B)
    e_p = 0

    # UPPER_ANGLE = 30
    # arg = np.sin(np.deg2rad(UPPER_ANGLE))
    i_b = np.pi/2 - np.arcsin(rng.uniform(0, 0.5))
    omega_b = rng.uniform(0, 2*np.pi)
    Omega_b = rng.uniform(0, 2*np.pi)
    omega_p = rng.uniform(0, 2*np.pi)
    Omega_p = rng.uniform(0, 2*np.pi)
    di = rng.rayleigh(np.deg2rad(sigma_di))
    i_p = i_b+di
    i_p_sky, Omega_p_sky, omega_p_sky = sky_transform(i_b, Omega_b, omega_b,
                                                      di, Omega_p, omega_p)

    # Simulate Orbit
    sim = init(M_A, M_B, a_b_circ, a_p, inc_bin=i_b, inc_p=i_p,
               e_B=e_b_circ, e_p=e_p, Omega_bin=Omega_b, omega_b=omega_b,
               omega_p=omega_p_sky, Omega_p=Omega_p_sky)
    sim.dt, BASELINE = 0.01, 8

    x, y, z = integrate(sim, BASELINE)
    b = sim.particles[1]
    p = sim.particles[2]

    # max_orbit = np.nanmax(np.sqrt(x[2]**2+y[2]**2+z[2]**2))/a_p
    P_final = p.P

    N = BASELINE/P_final
    # if max_orbit > 10 or P_final < 0: # unstable orbit
    #     system = np.array([P_b, P_b_circ, None, a_b_circ, 
    #                        P_p, None, a_p, M_A, M_B, R_A, 
    #                         R_B, e_b, e_b_circ, None,
    #                         None, i_b, i_p, i_p_sky,
    #                         Omega_b, Omega_p, Omega_p_sky, 
    #                         omega_b, omega_p, omega_p_sky,
    #                         None, -1])
    #     return system
    
    transits = check_transit(R_A, R_B, R_p, x, y, z)
    system = np.array([P_b, P_b_circ, b.P*yr, a_b_circ,
                        P_p, P_final*yr, a_p,
                        M_A, M_B, R_A, R_B,
                        e_b, e_b_circ, b.e, p.e,
                        i_b, i_p, i_p_sky,
                        Omega_b, Omega_p, Omega_p_sky, 
                        omega_b, omega_p, omega_p_sky,
                        N, transits])

    return system

def wrapper(params: list):

    if len(params) != 4:
        print("Bad Parameters")
        return
   
    return sample_check(*params)