#!/usr/bin/python3
"""
Copyright 2016-2017 Daniel Kolosa
This program is distributed under the GPL, for more information
refer to LICENSE.txt

perform ASE targeting as an LQ optimal control algorithm
"""

import matplotlib.pyplot as plt
import mpmath
from scipy.integrate import odeint
import time
import math

# Custom modules
from Mathmodels import *
from Orbit_Transformations import *

# Constants
RE = 6378
mu = 398600 * 60 ** 4 / RE ** 3
sec_to_hr = 60**2


def main():

    # Initial Orbit State
    a0 = 6500 / RE  # km
    e0 = 0.1
    i0 = math.radians(1.0) # rad
    Omega0 = math.radians(20.0)  # rad
    w0 = math.radians(15.0)  # rad
    M0 = math.radians(15.0) # rad

    # Initial Conditions
    x0 = np.array([[a0], [e0], [i0], [Omega0], [w0], [M0]])

    # transfer time
    t0 = 0.0
    ttarg = 2 * math.pi * math.sqrt(a0 ** 3 / mu) * 5
    # ttarg = 24*60**2 * 2 / 60**2
    dt = ttarg / 500
    tspan = np.arange(0.0, ttarg, dt)

    # target orbit state
    atarg = 7100 /RE  #7100/RE
    etarg = 0.2
    itarg = math.radians(1.1)
    Omegatarg = math.radians(30.0)
    wtarg = math.radians(25.0)
    Mtarg = math.radians(25.0)

    # convert orbital elements to Cartesian
    [r0, v0] = oe_to_rv(x0, t0)
    
    xT = np.array([[atarg], [etarg], [itarg], 
                   [Omegatarg], [wtarg], [Mtarg]])
    icl = x0 - xT
    Kf = 100 * np.eye(6)
    Q = 0.1 * np.eye(6)
    Q[5, 5] = 0.0
    R = 10 * np.eye(14)

    # Low Fidelity Model Parameters
    A = np.zeros((6, 6))
    B = find_G_M(a0, e0, i0, w0)
    u = np.zeros((len(tspan), 14))

    # optimize LF model
    yol = np.append(icl, 0)
    Pbig = odeint(findP, Kf.flatten(), tspan[::-1], args=(A, B, R, Q))
    Pbig = np.flipud(Pbig)

    Yl = odeint(ASE, yol, tspan, args=(A, B, R, Q, xT, Pbig, tspan))

    al = Yl[:,0] + atarg
    el = Yl[:,1] + etarg
    il = Yl[:,2] + itarg
    Omegal = Yl[:,3] + Omegatarg
    wl = Yl[:,4] + wtarg
    Ml = np.zeros((len(tspan), 1))

    rl, vl = (np.zeros((len(al), 3)) for _ in range(2))
    Jl = Yl[:,6]

    nt = np.sqrt(mu/al**3)
    Ml = Yl[:,5] + Mtarg+nt*tspan
    Ml %= 2*np.pi
    xl = np.column_stack((al, el, il, Omegal, wl, Ml))

    for j in range(1, len(tspan)):
        [rl[j,:], vl[j,:]] = oe_to_rv(xl[j,:], tspan[j])

    old_err_state = np.seterr(divide='raise')
    rcv = np.cross(rl, vl, axis=1)
    rhat = np.true_divide(rl, np.reshape(np.linalg.norm(rl,axis=1), (rl.shape[0],1)))
    what = np.true_divide(rcv, np.reshape(np.linalg.norm(rcv,axis=1), (rcv.shape[0],1)))
    shat = np.cross(what,rhat, axis=1)
    fthat = rhat + what + shat
    that = np.true_divide(fthat, np.reshape(np.linalg.norm(fthat, axis=1), (fthat.shape[0],1)))

    E, FR, FS, FW, FT = (np.zeros(len(tspan)) for _ in range(5))

    # Compute the Input vector
    for j in range(len(tspan)):
        Pvec = Pbig[j,:]
        P = np.reshape(Pvec, (6,6))
        u[j,:] = -np.linalg.inv(R) @ B.T @ P @ Yl[j,0:6]
        E[j] = kepler([al[j], el[j], il[j], Omegal[j], wl[j], Ml[j]], tspan[j])

        # Compute the Thrust Fourier Coefficients
    FR = u[:,0] + u[:,1] * np.cos(E) + u[:,2] * np.cos(2*E) + u[:,3] * np.sin(E)
    FS = u[:,4] + u[:,5] * np.cos(E) + u[:,6] * np.cos(2*E) + u[:,7] * np.sin(E) + u[:,8]*np.sin(2*E)
    FW = u[:,9] + u[:,10] * np.cos(E) + u[:,11] * np.cos(2*E) + u[:,12] * np.sin(E) + u[:,13]*np.sin(2*E)
    FT = np.sqrt(FR**2 + FS**2 + FW**2)
    conv_to_mN = RE*1000/60**4*1000
    FR *= conv_to_mN
    FW *= conv_to_mN
    FS *= conv_to_mN
    FT *= conv_to_mN
    # Use Newton's EOM
    y0Newt = np.concatenate((r0, v0, [0])).flatten()  # [r, v, F_t]
    YNewt = odeint(Newt_EOM, y0Newt, tspan, args=(u, tspan))

    aNewt, eNewt, iNewt, OmegaNewt, wNewt, thetaNewt, ENewt, MNewt = (np.zeros(len(tspan)) for _ in range(8))

    for j in range(len(tspan)):
        [aNewt[j], eNewt[j], iNewt[j], OmegaNewt[j], wNewt[j], thetaNewt[j], ENewt[j], MNewt[j]] = rv_to_oe(YNewt[j, 0:3], YNewt[j, 3:6])


    generate_plots(tspan, xl,
                   aNewt, eNewt, iNewt, OmegaNewt, wNewt, MNewt,
                   atarg, etarg, itarg, Omegatarg, wtarg, Mtarg,
                   FR, FS, FW,
                   YNewt, Yl)

def find_G_M(a, e, i, w):
    """ Use Gaussian equations to compute inputs  """

    G = np.zeros((6, 14))

    # alpha = [a0R a1R a2R b1R a0S a1S a2S b1S b2S a0W a1W a2W b1W b2W]'; %RSW

    #a
    G[0, 3] = np.sqrt(a**3/mu)*e #b1R
    G[0, 4] = 2*np.sqrt(a**3/mu)*np.sqrt(1-e**2) #a0S

    #e
    G[1, 3] = .5*np.sqrt(1-e**2) #b1R
    G[1, 4] = -1.5*e  # a0S
    G[1, 5] = 1  # a1S
    G[1, 6] = -.25*e  # a2S
    G[1, :] = G[1,:]*np.sqrt(a/mu)*np.sqrt(1-e**2)

    #i
    G[2, 9] = -1.5*e*np.cos(w)  # a0W
    G[2, 10] = .5*(1+e**2)*np.cos(w)  # a1W
    G[2, 11] = -.25*e*np.cos(w)  # a2W
    G[2, 12] = -.5*np.sqrt(1-e**2)*np.sin(w)  # b1W
    G[2, 13] = .25*e*np.sqrt(1-e**2)*np.sin(w)  # b2W
    G[2, :] = G[2, :]*np.sqrt(a/mu)/np.sqrt(1-e**2)

    #Omega
    G[3, 9] = -1.5*e*np.sin(w)  # a0W
    G[3, 10] = 0.5*(1+e**2)*np.sin(w)  # a1W
    G[3, 11] = -0.25*e*np.sin(w)  # a2W
    G[3, 12] = 0.5*np.sqrt(1-e**2)*np.cos(w)  # b1W
    G[3, 13] = -0.25*e*np.sqrt(1-e**2)*np.cos(w)  # b2W
    G[3, :] = G[3, :]*np.sqrt(a/mu)*mpmath.csc(i)/np.sqrt(1-e**2)

    #w
    G[4, 0] = e*np.sqrt(1-e**2)  # a0R
    G[4, 1] = -.5*np.sqrt(1-e**2)  # a1R
    G[4, 7] = .5*(2-e**2)  # b1S
    G[4, 8] = -.25*e  # b2S
    G[4, :] = G[4, :]*np.sqrt(a/mu)/e
    G[4, :] = G[4, :] - np.cos(i)*G[3,:]

    #M
    G[5, 0] = -2-e**2  # a0R
    G[5, 1] = 2*e  # a1R
    G[5, 3] = -.5*e**2  # a2R
    G[5, :] = G[5,:]*np.sqrt(a/mu)
    G[5, :] = G[5,:]+(1 - np.sqrt(1-e**2)) * (G[4,:]+G[3,:])+\
              2*np.sqrt(1-e**2)*(np.sin(i/2))**2*G[3,:]-(G[4,:]+G[3,:])

    return G


def findP(Pvec, t, A, B, R, Q):
    """ Differential Ricatti Equation """
    P = np.zeros((6, 6))
    P = np.reshape(Pvec, (6, 6))
    # P[:] = Pvec
    Pdot = -(A.T.dot(P) + P.dot(A) - P.dot(B).dot(np.linalg.inv(R)).dot(B.T).dot(P) + Q)
    return Pdot.flatten()


def generate_plots(tspan, xl,
                   aNewt, eNewt, iNewt, OmegaNewt, wNewt, MNewt,
                   atarg, etarg, itarg, Omegatarg, wtarg, Mtarg,
                   FR, FS, FW,
                   YNewt, Yl):

    plt.figure(figsize=(8,6))
    plt.subplot(3, 2, 1)
    plt.plot(tspan[-1], atarg, 'ro')
    plt.plot(tspan, xl[:,0])
    plt.plot(tspan, aNewt)
    plt.ylabel('a')

    plt.subplot(3, 2, 2)
    plt.plot(tspan[-1], etarg, 'ro')
    plt.plot(tspan, xl[:,1])
    plt.plot(tspan, eNewt)
    plt.ylabel('e')

    plt.subplot(3, 2, 3)
    plt.plot(tspan[-1], itarg, 'ro')
    plt.plot(tspan, xl[:,2])
    plt.plot(tspan, iNewt)
    plt.ylabel('i')

    plt.subplot(3, 2, 4)
    plt.plot(tspan[-1], Omegatarg, 'ro')
    plt.plot(tspan, xl[:,3])
    plt.plot(tspan, OmegaNewt)
    plt.ylabel('$\Omega$')

    plt.subplot(3, 2, 5)
    plt.plot(tspan[-1], wtarg, 'ro')
    plt.plot(tspan, xl[:,4])
    plt.plot(tspan, wNewt)
    plt.ylabel('$\omega$')

    plt.subplot(3, 2, 6)
    plt.plot(tspan[-1], Mtarg, 'ro')
    plt.plot(tspan, xl[:,5])
    plt.plot(tspan, MNewt)
    plt.ylabel('M')

    plt.figure(2)
    FRplot = plt.plot(tspan, FR, label='FR')
    FSplot = plt.plot(tspan, FS, label='FS')
    FWplot = plt.plot(tspan, FW, label='FW')
    plt.xlabel('time(hr)')
    plt.ylabel('mN')
    plt.legend()

    # plot the x, y, and z in 3d plot
    # plt.figure(3)
    # plt.axes(projection='3d')
    # plt.plot(YNewt[:,0], YNewt[:,1], YNewt[:,2])
    # plt.plot(Yl[:,0], Yl[:,1], Yl[:,2])

    plt.show()

    
if __name__ == '__main__':
    main()
