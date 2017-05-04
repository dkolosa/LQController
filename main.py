#!/usr/bin/python3

#Copyright 2015-2016 Daniel Kolosa
#This program is distributed under the GPL, for more information
#refer to LICENSE.txt

#perform ASE targeting as an LQ optimal control algorithm

import numpy as np
from scipy.integrate import odeint
import mpmath


def main():
    # Constants
    global mu, A, B, R, Q
    RE = 6378
    deg_to_rad = np.pi/180
    mu = 398600 * 60 ** 4 / RE ** 3
    
    # Initial Orbit State
    a0 = 6678 / RE  # km
    e0 = 0.67
    i0 = 20 * deg_to_rad # rad
    Omega0 = 20 * deg_to_rad  # rad
    w0 = 20 * deg_to_rad # rad
    M0 = 20 * deg_to_rad  # rad

    n0 = np.sqrt(mu / a0 ** 3) # Mean motion

    # Initial COnditions
    x0 = np.array([[a0], [e0], [i0], [Omega0], [w0], [M0]])

    # transfer time
    t0 = 0
    ttarg = 2 * np.pi * np.sqrt(a0 ** 3 / mu) * 20
    dt = ttarg / 500
    tspan = np.arange(t0, ttarg, dt)
    tspan_bk = tspan[::-1]

    # convert orbital elements to Cartesian
    [r0, v0] = oe_to_rv(x0, t0)

    # target orbit state
    atarg = 7345.0/RE
    etarg = .67
    itarg = 10.0 * deg_to_rad
    Omegatarg = 20.0 * deg_to_rad
    wtarg = 20.0 * deg_to_rad
    Mtarg = 20.0 * deg_to_rad
    xT = np.array([[atarg], [etarg], [itarg], 
                   [Omegatarg], [wtarg], [Mtarg]])

    icl = x0-xT
    Kf = 100*np.eye(6)
    Q = .1*np.eye(6)
    Q[5,5] = 0
    R = 1*np.eye(14)

    # LF Model Parameters
    A = np.zeros((6,6))
    B = find_G_M(a0, e0, i0, w0)
    u = np.zeros((len(tspan),14))

    # optimize LF model
    yol = [[icl],[0]]
    Pbig = odeint(findP, Kf[:], tspan_bk)

    Yl = odeint(ASE, yol, tspan)

    al = Yl[:,0]+atarg
    el = Yl[:,1]+etarg
    il = Yl[:,2]+itarg
    Omegal = Yl[:,3]+Omegatarg
    wl = Yl[:,4]+wtarg
    Ml = np.zeros(len(tspan),1)
    rl = np.zeros(len(al),3)
    vl = np.zeros(len(al),3)

    for j in range(1,len(tspan)):
        nt=np.sqrt(mu/al[j]**3)
        Ml[j] = Yl[j,5]+Mtarg+nt*tspan[j]
        Ml[j] %= (2*np.pi)
        xl = np.array([al[j], el[j], il[j], Omegal[j], wl[j], Ml[j]])
        # Convert orbital elements to Cartesian
        [rl[j,0:2],vl[j,0:2]] = oe_to_rv(xl,tspan[j]) 

    #calculate the direction in terms of the x,y,z
    rhat = np.zeros(len(tspan),3)
    what = np.zeros(len(tspan),3)
    shat = np.zeros(len(tspan),3)
    that = np.zeros(len(tspan),3)

    for k in range(1,len(rl))
        rcv = np.cross(rl[k,:],vl[k,:])
        rhat[k,:] = rl[k,:]/np.linalg.norm(rl[k,:])
        what[k,:] = rcv/np.linalg.norm(rcv)
        shat[k,:] = np.cross(what[k,:],rhat[k,:])
        fthat = rhat[k,:] + what[k,:] + shat[k,:]
        that[k,:] = fthat/np.linalg.norm(fthat)


    E = np.zeros(range(tspan))
    FR = np.zeros(len(tspan))
    FS = np.zeros(len(tspan))
    FW = np.zeros(len(tspan))
    FT = np.zeros(len(tspan))

    # COmpute the Input vector
    for j in range(len(tspan)):
        Pvec = Pbig[j,:]
        P = np.zeros(6, 6)
        P[:] = Pvec
        u[j,:] = np.dot(np.dot(np.dot(-np.inv(R),B.T),P), Yl[j,0:5]).T
        E[j] = kepler([al[j], el[j], il[j], Omegal[j], wl[j], Ml[j]], tspan)

        # Compute the Thrust Fourier Coefficients
        FR[j] = u[j,0] + u[j,1] * np.cos(E[j]) + u[j,2] * np.cos(2*E[j]) + u[j,3] * np.sin(E[j])
        FS[j] = u[j,4] + u[j,5] * np.cos(E[j]) + u[j,6] * np.cos(2*E[j]) + u[j,7] * np.sin(E[j]) + u[j,8]*np.sin(2*E[j])
        FW[j] = u[j,9] + u[j,10] * np.cos(E[j]) + u[j,11] * np.cos(2*E[j]) + u[j,12] * np.sin(E[j]) + u[j,14]*np.sin(2*E[j])
        FT[j] = np.sqrt(FR[j]**2 + Fs[j]**2 + FW[j]**2)


    # Use Newton's EOM 
    y0Newt = [r0, v0, 0]
    YNewt = odeint('Newt_EOM', y0Newt, tspan)

    aNewt = np.zeros(len(tspan), 1)
    eNewt = np.zeros(len(tspan), 1)
    iNewt = np.zeros(len(tspan), 1)
    OmegaNewt = np.zeros(len(tspan), 1)
    wNewt = np.zeros(len(tspan), 1)
    thetaNewt = np.zeros(len(tspan), 1)
    ENewt = np.zeros(len(tspan), 1)
    MNewt = np.zeros(len(tspan), 1)

    for j in len(tspan):
        [aNewt[j],eNewt[j],iNewt[j],OmegaNewt[j],wNewt[j],thetaNewt[j],ENewt[j[,MNewt[j]] = rv_to_oe(YNewt[j,1:3],YNewt[j,4:6])



def oe_to_rv(oe,t):
    a=oe[0]
    e=oe[1]
    i=oe[2]
    Omega=oe[3]
    w=oe[4]
    M=oe[5]
    #%Determine orbit type
    if a<0 or e<0 or e>1 or math.fabs(i)>2*math.pi or math.fabs(Omega)>2*math.pi or math.fabs(w)>2*math.pi: #problem
        print(a)
        print(e)
        print(i)
        print(Omega)
        print(w)
        print('Invalid orbital element(s)')
    xhat=np.array([1, 0, 0])
    yhat=np.array([0, 1, 0])
    zhat=np.array([0, 0, 1])

    nu=kepler(oe, t)
    nhat=math.cos(Omega)*xhat+math.sin(Omega)*yhat
    # hhat=sin(i)*sin(Omega)*xhat-sin(i)*cos(Omega)*yhat+cos(i)*zhat
    rhatT=-math.cos(i)*math.sin(Omega)*xhat+math.cos(i)*math.cos(Omega)*yhat+math.sin(i)*zhat
    rmag=a*(1-e**2)/(1+e*math.cos(nu))
    vmag=math.sqrt(mu/rmag*(2-rmag/a))
    gamma=math.atan2(e*math.sin(nu),1+e*math.cos(nu))
    u=w+nu
    rhat=math.cos(u)*nhat+math.sin(u)*rhatT
    vhat=math.sin(gamma-u)*nhat+math.cos(gamma-u)*rhatT
    r=rmag*rhat
    v=vmag*vhat

    return np.array([r, v])


def kepler(oe,t):
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
    if math.fabs(math.sin(Mstar))>1e-10: #check that nu~=0
        sigma = math.sin(Mstar)/math.fabs(math.sin(Mstar))    #sgn(sin(Mstar))
        x = Mstar+sigma*k*e
        for count in range(1,10):
            es = e*math.sin(x)
            f = x-es-Mstar
            if math.fabs(f) < delta:
                E = x
                nu = 2*math.atan2(math.sqrt((1+e)/(1-e))*math.tan(E/2),1)
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
    #       else %test that computations were correct
    #       time=(E-e*math.sin(E))/math.sqrt(mu/a**3)+Tau

    else:
        nu = 0
        E = 0
    #time=(E-e*math.sin(E))/math.sqrt(mu/a**3)+Tau

    return nu


def find_G_M(a,e,i,w):
    """ Use Gaussian equations to compute inputs  """

    G = np.zeros((6,14))

    #alpha = [a0R a1R a2R b1R a0S a1S a2S b1S b2S a0W a1W a2W b1W b2W]'; %RSW

    #a
    G[0,3]=np.sqrt(a**3/mu)*e #b1R
    G[0,4]=2*np.sqrt(a**3/mu)*np.sqrt(1-e**2) #a0S


    #e
    G[1,3]=.5*np.sqrt(1-e**2) #b1R
    G[1,4]=-1.5*e #a0S
    G[1,5]=1 #a1S
    G[1,6]=-.25*e #a2S
    G[1,:]=G[1,:]*np.sqrt(a/mu)*np.sqrt(1-e**2)

    #i
    G[2,9]=-1.5*e*np.cos(w) #a0W
    G[2,10]=.5*(1+e**2)*np.cos(w) #a1W
    G[2,11]=-.25*e*np.cos(w) #a2W
    G[2,12]=-.5*np.sqrt(1-e**2)*np.sin(w) #b1W
    G[2,13]=.25*e*np.sqrt(1-e**2)*np.sin(w) #b2W
    G[2,:]=G[2,:]*np.sqrt(a/mu)/np.sqrt(1-e**2)

    #Omega
    G[3,9]=-1.5*e*np.sin(w) #a0W
    G[3,10]=.5*(1+e**2)*np.sin(w) #a1W
    G[3,11]=-.25*e*np.sin(w) #a2W
    G[3,12]=.5*np.sqrt(1-e**2)*np.cos(w) #b1W
    G[3,13]=-.25*e*np.sqrt(1-e**2)*np.cos(w) #b2W
    G[3,:]=G[3,:]*np.sqrt(a/mu)*mpmath.csc(i)/np.sqrt(1-e**2)

    #w
    G[4,0]=e*np.sqrt(1-e**2) #a0R
    G[4,1]=-.5*np.sqrt(1-e**2) #a1R
    G[4,7]=.5*(2-e**2) #b1S
    G[4,8]=-.25*e #b2S
    G[4,:]=G[4,:]*np.sqrt(a/mu)/e
    G[4,:]=G[4,:]-np.cos(i)*G[3,:]

    #M
    G[5,0]=-2-e**2 #a0R
    G[5,1]=2*e #a1R
    G[5,3]=-.5*e**2 #a2R
    G[5,:]=G[5,:]*np.sqrt(a/mu)
    G[5,:]=G[5,:]+(1 - np.sqrt(1-e**2)) * (G[4,:]+G[3,:])+\
           2*np.sqrt(1-e**2)*(np.sin(i/2))**2*G[3,:]-(G[4,:]+G[3,:])

    return G


def findP(t,Pvec):
    """ Ricatti Equation """
    P = np.zeros(6)
    P[:] = Pvec
    Pdot = -(np.transpose(A)*P+P*A-P*B*(R**-1)*np.transpose(B)*P+Q)

    return Pdot[:]


def ASE(t,y):
    x = y[1:6]+xT
    Pvec = scipy.interp(t, tspan, Pbig)

    P=np.zeros(6)
    P[:]=Pvec

    nt=np.sqrt(mu/x[0]**3)
    F=np.array([np.zeros((5,1)), [nt]])

    u_t = -(R**-1)*np.transpose(B)*P*y[0:5]

    dx = A*x + B*u_t
    dJ = np.transpose(y[0:5])*Q*y[0:5] + np.transpose(u_t)*R*u_t

    dxl = [[dx],[dJ]]

    return dxl


if __name__=='__main__':
    main()





