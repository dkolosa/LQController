
import numpy as np
import math

RE = 6378
mu = 398600 * 60 ** 4 / RE ** 3


def oe_to_rv(oe, t):
    """ Convert the orbital elements to Cartesian """
    a, e, i, Omega, w, M = oe[0], oe[1], oe[2], oe[3], oe[4], oe[5]

    # Determine orbit type
    if a < 0 or e < 0 or e > 1 or abs(i) > 2*np.pi or abs(Omega)>2*np.pi or abs(w)>2*np.pi:  # problem
        print(a)
        print(e)
        print(i)
        print(Omega)
        print(w)
        print('Invalid orbital element(s)')
    xhat = np.array([1, 0, 0])
    yhat = np.array([0, 1, 0])
    zhat = np.array([0, 0, 1])

    nu = kepler(oe, t)
    nhat = np.cos(Omega)*xhat+np.sin(Omega)*yhat
    rhatT = -np.cos(i)*np.sin(Omega)*xhat + np.cos(i)*np.cos(Omega)*yhat + np.sin(i)*zhat
    rmag = a*(1-e**2)/(1+e*np.cos(nu))
    vmag = np.sqrt(mu/rmag*(2-rmag/a))
    gamma = np.arctan2(e*np.sin(nu), 1+e*np.cos(nu))
    u = w + nu
    rhat = np.cos(u)*nhat + np.sin(u)*rhatT
    vhat = np.sin(gamma-u)*nhat + np.cos(gamma-u)*rhatT
    r = rmag*rhat
    v = vmag*vhat

    return [r, v]


def rv_to_oe(r, v):

    r_hat = r / np.linalg.norm(r)
    h = np.cross(r, v)
    h_hat = h / np.linalg.norm(h)

    z = np.array([0, 0, 1])
    n = np.cross(z, h)

    if np.linalg.norm(n) < 1e-8:
        e_vec = np.cross(v, h)/mu - r_hat
        energy = 0.5 * np.dot(v,v) - mu/np.linalg.norm(r)
        a = -m / (2*energy)
        e = np.linalg.norm(e_vec)
        i, Omega = 0, 0
        if abs(e) < 1e-8:
            w = 0
        else:
            w = np.arccos(e_vec[0] / e)
    else:
        n_hat = n / np.linalg.norm(n)
        e_vec = np.cross(v, h)/mu - r_hat
        energy = .5*np.dot(v,v) - mu/np.linalg.norm(r)
        a = -mu / (2*energy)
        e = np.linalg.norm(e_vec)
        i = np.arccos(np.dot(z,h_hat))
        Omega = np.arctan2(n_hat[1], n_hat[0])
        if abs(e) < 1e-8:
            w = 0
        else:
            w = np.arctan2(np.dot(h_hat, np.cross(n_hat, e_vec)),
                           np.dot(n_hat, e_vec))

    if w < 0:
        w += 2*np.pi
    if Omega < 0:
        Omega += 2*np.pi

    theta = np.arctan2(np.dot(h_hat, np.cross(e_vec, r)), np.dot(e_vec, r))
    E = 2*math.atan2(math.sqrt(1-e) * math.tan(theta/2), math.sqrt(1+e))

    if E < 0:
        E += 2*math.pi

    M = E - e*math.sin(E)
    if M < 0:
        M += 2*math.pi

    return [a, e, i, Omega, w, theta, E, M]


def kepler(oe, t):
    """ Using kepler's equations to calculate True Anomaly"""
    a = oe[0]
    e = oe[1]
    i = oe[2]
    omega = oe[3]
    w = oe[4]
    M = oe[5]
    dx = 0

    #Calculate True anamoly
    k = .85
    delta = 1e-14
    Mstar = M-math.floor(M/(2*math.pi))*2*math.pi
    if abs(math.sin(Mstar)) > 1e-10: #check that nu~=0
        sigma = math.sin(Mstar)/abs(math.sin(Mstar))    #sgn(sin(Mstar))
        x = Mstar + sigma*k*e
        for count in range(0, 10):
            es = e*math.sin(x)
            f = x-es-Mstar
            if abs(f) < delta:
                E = x
                nu = 2*math.atan2(math.sqrt((1+e)/(1-e))*math.tan(E/2), 1)
                break
            else:
                ec = e*math.cos(x)
                fp = 1-ec
                fpp = es
                fppp = ec
                dx = -f/(fp+dx*fpp/2+dx**2*fppp/6)
                x = x+dx

        if count == 10: #check that Newton's method converges
            nu = 'undefined'
    else:
        nu = 0
        E = 0

    return E
