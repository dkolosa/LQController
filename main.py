#!/usr/bin/python3
"""
Copyright 2016-2017 Daniel Kolosa
This program is distributed under the GPL, for more information
refer to LICENSE.txt

perform ASE targeting as an LQ optimal control algorithm
"""

import numpy as np
from scipy.integrate import odeint
import mpmath
import matplotlib.pyplot as plt
from Mathmodels import *

import time

RE = 6378
mu = 398600 * 60 ** 4 / RE ** 3


def main():

    # Constants
    deg_to_rad = np.pi/180

    # Initial Orbit State
    a0 = 6678 / RE  # km
    e0 = 0.67
    i0 = 20.0 * deg_to_rad # rad
    Omega0 = 20.0 * deg_to_rad  # rad
    w0 = 20.0 * deg_to_rad # rad
    M0 = 20.0 * deg_to_rad  # rad

    n0 = np.sqrt(mu / a0 ** 3)  # Mean motion

    # Initial Conditions
    x0 = np.array([[a0], [e0], [i0], [Omega0], [w0], [M0]])

    # transfer time
    t0 = 0
    ttarg = 2 * np.pi * np.sqrt(a0 ** 3 / mu) * 1
    dt = ttarg / 100
    tspan = np.arange(0, ttarg, dt)
    tspan_bk = tspan[::-1]

    # convert orbital elements to Cartesian
    [r0, v0] = oe_to_rv(x0, t0)

    # target orbit state
    atarg = 1.1517  #7100/RE
    etarg = .7370
    itarg = 10.0 * deg_to_rad
    Omegatarg = 20.0 * deg_to_rad
    wtarg = 20.0 * deg_to_rad
    Mtarg = 20.0 * deg_to_rad
    xT = np.array([[atarg], [etarg], [itarg], 
                   [Omegatarg], [wtarg], [Mtarg]])

    icl = x0-xT
    Kf = 100*np.eye(6)
    Q = 0.1*np.eye(6)
    Q[5, 5] = 0.0
    R = 1*np.eye(14)

    # Low Fidelity Model Parameters
    A = np.zeros((6, 6))
    B = find_G_M(a0, e0, i0, w0)
    u = np.zeros((len(tspan), 14))

    # optimize LF model
    yol = np.append(icl, 0)
    Pbig = odeint(findP, Kf.flatten(), tspan_bk, args=(A, B, R, Q))
    Pbig = Pbig[::-1]

    Yl = odeint(ASE, yol, tspan, args=(A, B, R, Q, xT, Pbig, tspan))

    al = Yl[:,0] + atarg
    el = Yl[:,1] + etarg
    il = Yl[:,2] + itarg
    Omegal = Yl[:,3] + Omegatarg
    wl = Yl[:,4] + wtarg
    Ml = np.zeros((len(tspan), 1))

    rl = np.zeros((len(al), 3))
    vl = np.zeros((len(al), 3))
    Jl = Yl[:,6]

    for j in range(1, len(tspan)):
        nt=np.sqrt(mu/al[j]**3)
        Ml[j] = Yl[j,5] + Mtarg+nt*tspan[j]
        Ml[j] %= (2*np.pi)
        xl = np.array([al[j], el[j], il[j], Omegal[j], wl[j], Ml[j]])
        # Convert orbital elements to Cartesian
        [rl[j, 0:3], vl[j, 0:3]] = oe_to_rv(xl, tspan[j])

    # #calculate the direction in terms of the x,y,z
    # rhat = np.zeros((len(tspan), 3))
    # what = np.zeros((len(tspan), 3))
    # shat = np.zeros((len(tspan), 3))
    # that = np.zeros((len(tspan), 3))
    #
    # # old_err_state = np.seterr(divide='raise')
    # # ignored_states = np.seterr(**old_err_state)
    # # for k in range(len(rl)):
    # #     rcv = np.cross(rl[k,:],vl[k,:])
    # #     rhat[k,:] = np.true_divide(rl[k,:], np.linalg.norm(rl[k,:]))
    # #     what[k,:] = np.true_divide(rcv, np.linalg.norm(rcv))
    # #     shat[k,:] = np.cross(what[k,:],rhat[k,:])
    # #     fthat = rhat[k,:] + what[k,:] + shat[k,:]
    # #     that[k,:] = np.true_divide(fthat, np.linalg.norm(fthat))

    E = np.zeros(len(tspan))
    FR = np.zeros(len(tspan))
    FS = np.zeros(len(tspan))
    FW = np.zeros(len(tspan))
    FT = np.zeros(len(tspan))

    # Compute the Input vector
    start = time.time()
    for j in range(len(tspan)):
        Pvec = Pbig[j,:]
        P = np.reshape(Pvec, (6,6))
        u[j,:] = -np.linalg.inv(R).dot(B.T).dot(P).dot(Yl[j,0:6]).T
        E[j] = kepler([al[j], el[j], il[j], Omegal[j], wl[j], Ml[j]], tspan)

        # Compute the Thrust Fourier Coefficients
        FR[j] = u[j,0] + u[j,1] * np.cos(E[j]) + u[j,2] * np.cos(2*E[j]) + u[j,3] * np.sin(E[j])
        FS[j] = u[j,4] + u[j,5] * np.cos(E[j]) + u[j,6] * np.cos(2*E[j]) + u[j,7] * np.sin(E[j]) + u[j,8]*np.sin(2*E[j])
        FW[j] = u[j,9] + u[j,10] * np.cos(E[j]) + u[j,11] * np.cos(2*E[j]) + u[j,12] * np.sin(E[j]) + u[j,13]*np.sin(2*E[j])
        FT[j] = np.sqrt(FR[j]**2 + FS[j]**2 + FW[j]**2)
    end = time.time()
    print(end-start)
    # Use Newton's EOM
    y0Newt = np.concatenate((r0, v0, [0]))  # [r, v, F_t]
    YNewt = odeint(Newt_EOM, y0Newt, tspan, args=(u, tspan))

    aNewt = np.zeros(len(tspan))
    eNewt = np.zeros(len(tspan))
    iNewt = np.zeros(len(tspan))
    OmegaNewt = np.zeros(len(tspan))
    wNewt = np.zeros(len(tspan))
    thetaNewt = np.zeros(len(tspan))
    ENewt = np.zeros(len(tspan))
    MNewt = np.zeros(len(tspan))

    for j in range(len(tspan)):
        [aNewt[j], eNewt[j], iNewt[j], OmegaNewt[j], wNewt[j], thetaNewt[j], ENewt[j], MNewt[j]] = rv_to_oe(YNewt[j, 0:3], YNewt[j, 3:6])

    rad_to_deg = 180/np.pi
    plt.figure()
    plt.subplot(3, 2, 1)
    plt.plot(tspan, al*RE)
    plt.ylabel('a')
    plt.subplot(3, 2, 2)
    plt.plot(tspan, el)
    plt.ylabel('e')
    plt.subplot(3, 2, 3)
    plt.plot(tspan, il*rad_to_deg)
    plt.ylabel('i')
    plt.subplot(3, 2, 4)
    plt.plot(tspan, Omegal*rad_to_deg)
    plt.ylabel('$\Omega$')
    plt.subplot(3, 2, 5)
    plt.plot(tspan, wl*rad_to_deg)
    plt.ylabel('$\omega$')
    plt.subplot(3, 2, 6)
    plt.plot(tspan, Ml*rad_to_deg)
    plt.ylabel('M')

    plt.figure(2)
    FRplot = plt.plot(tspan, FR, label='FR')
    FSplot = plt.plot(tspan, FS, label='FS')
    FWplot = plt.plot(tspan, FW, label='FW')
    plt.legend()

    # plot the x, y, and z
    plt.figure(3)
    xyplt = plt.plot(rl[:, 0] * RE, rl[:, 1]*RE, label='xy')
    plt.xlabel('x')
    plt.ylabel('y')

    # plt.show()


def oe_to_rv(oe, t):
    """ Convert the orbital elements to Cartesian """
    a, e, i, Omega, w, M = oe[0], oe[1], oe[2], oe[3], oe[4], oe[5]

    # Determine orbit type
    if a < 0 or e < 0 or e>1 or abs(i) > 2*np.pi or abs(Omega)>2*np.pi or abs(w)>2*np.pi:  # problem
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
    # hhat=sin(i)*sin(Omega)*xhat-sin(i)*cos(Omega)*yhat+cos(i)*zhat
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

    z = [0, 0, 1]
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
    E = 2*np.arctan2(np.sqrt(1-e) * np.tan(theta/2), np.sqrt(1+e))

    if E < 0:
        E += 2*np.pi

    M = E - e*np.sin(E)
    if M < 0:
        M += 2*np.pi

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
    Mstar = M-np.floor(M/(2*np.pi))*2*np.pi
    if abs(np.sin(Mstar)) > 1e-10: #check that nu~=0
        sigma = np.sin(Mstar)/abs(np.sin(Mstar))    #sgn(sin(Mstar))
        x = Mstar + sigma*k*e
        for count in range(0, 11):
            es = e*np.sin(x)
            f = x-es-Mstar
            if abs(f) < delta:
                E = x
                nu = 2*np.arctan2(np.sqrt((1+e)/(1-e))*np.tan(E/2), 1)
                break
            else:
                ec = e*np.cos(x)
                fp = 1-ec
                fpp = es
                fppp = ec
                dx = -f/(fp+dx*fpp/2+dx**2*fppp/6)
                x = x+dx
        if count == 10: #check that Newton's method converges
            nu = 'undefined'
    #       else %test that computations were correct
    #       time=(E-e*math.sin(E))/math.sqrt(mu/a**3)+Tau

    else:
        nu = 0
        E = 0
    #time=(E-e*math.sin(E))/math.sqrt(mu/a**3)+Tau

    return nu


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

    P = np.reshape(Pvec, (6, 6))
    Pdot = -(A.T.dot(P) + P.dot(A) - P.dot(B).dot(np.linalg.inv(R)).dot(B.T).dot(P) + Q)
    return Pdot.flatten()


if __name__ == '__main__':
    main()





