#!/usr/bin/python3

#Copyright 2015-2016 Daniel Kolosa
#This program is distributed under the GPL, for more information
#refer to LICENSE.txt

#perform ASE targeting as an LQ optimal control algorithm

import numpy as np
from scipy.integrate import ode
import math
from math import pi


def main():
    # Constants
    global mu
    RE = 6378
    mu = 398600 * 60 ** 4 / RE ** 3
    # Initial Orbit State
    a0 = 6678 / RE  # km
    e0 = 0.67
    i0 = 20 * math.pi / 180  # rad
    Omega0 = 20 * math.pi / 180  # rad
    w0 = 20 * pi / 180  # rad
    M0 = 20 * pi / 180  # rad
    n0 = math.sqrt(mu / a0 ** 3)
    x0 = np.array([[a0], [e0], [i0], [Omega0], [w0], [M0]])
    # transfer time
    t0 = 0
    ttarg = 2 * pi * math.sqrt(a0 ** 3 / mu) * 20
    dt = ttarg / 500
    tspan = np.arange(t0, ttarg, dt)
    tspan_bk = tspan[::-1]
    # convert oe to rv
    [r0, v0] = oe_to_rv(x0, t0)

    #target state
    atarg = 7345/RE
    etarg = .67
    itarg = 10*math.pi/180
    Omegatarg = 20*math.pi/180
    wtarg = 20*math.pi/180
    Mtarg = 20*math.pi/180
    xT = np.array([[atarg], [etarg], [itarg], [Omegatarg], [wtarg], [Mtarg]])

    icl = x0-xT
    Kf = 100*np.eye(6)
    Q = .1*np.eye(6)
    Q[5,5] = 0
    R = 1*eye(14)

    #LF Model Parameters
    A = np.zeros((6,6))
    B = find_G_M(a0,e0,i0,w0)
    u = np.zeros((len(tspan),14))

    #optimize LF model
    yol=np.array([[icl],[0]])
    Pbig = ode(findP(A,B,Q,R),tspan_bk,Kf[:]).set_integrator(backend, nsteps=1)

    Yl = ode(ASE,tspan,yol).set_integrator(backend, nsteps=1)

    al = Yl[:,0]+atarg
    el = Yl[:,1]+etarg
    il = Yl[:,2]+itarg
    Omegal = Yl[:,3]+Omegatarg
    wl = Yl[:,4]+wtarg
    Ml=zeros(len(tspan),1)
    rl = np.zeros(len(al),3)
    vl = np.zeros(len(al),3)
    for j in range(1,len(tspan)):
        nt=sqrt(mu/al[j]**3)
        Ml[j] = Yl[j,5]+Mtarg+nt*tspan[j]
        Ml[j] %= (2*math.pi)
        xl = np.array([al[j], el[j], il[j], Omegal[j], wl[j], Ml[j]])
        [rl[j,0:2],vl[j,0:2]] = oe_to_rv(xl,tspan[j]) #convert orbital elements to r,v

    #calculate the direction in terms of the x,y,z
    rhat = np.zeros(len(tspan),3)
    what = np.zeros(len(tspan),3)
    shat = np.zeros(len(tspan),3)
    that = np.zeros(len(tspan),3)

    # for k in range(1,len(rl))
    #     rcv = cross(rl(k,:),vl(k,:))
    #     rhat(k,:) = rl(k,:)/norm(rl(k,:))
    #     what(k,:) = rcv/norm(rcv)
    #     shat(k,:) = cross(what(k,:),rhat(k,:))
    #     fthat = rhat(k,:) + what(k,:) + shat(k,:)
    #     that(k,:) = fthat/norm(fthat)




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
    vhat=sin(gamma-u)*nhat+cos(gamma-u)*rhatT
    r=rmag*rhat
    v=vmag*vhat

    return np.array([r, v])


def kepler(oe,t):
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
    #    else %test that computations were correct
    #       time=(E-e*math.sin(E))/math.sqrt(mu/a**3)+Tau
    else:
        nu = 0
        E = 0
    #time=(E-e*math.sin(E))/math.sqrt(mu/a**3)+Tau


def find_G_M(a,e,i,w):

    G = np.zeros((6,14))

    #alpha = [a0R a1R a2R b1R a0S a1S a2S b1S b2S a0W a1W a2W b1W b2W]'; %RSW

    #a
    G[0,3]=math.sqrt(a**3/mu)*e #b1R
    G[0,4]=2*math.sqrt(a**3/mu)*math.sqrt(1-e**2) #a0S


    #e
    G[1,3]=.5*math.sqrt(1-e**2) #b1R
    G[1,4]=-1.5*e #a0S
    G[1,5]=1 #a1S
    G[1,6]=-.25*e #a2S
    G[1,:]=G[1,:]*math.sqrt(a/mu)*math.sqrt(1-e**2)

    #i
    G[2,10]=-1.5*e*math.cos(w) #a0W
    G[2,11]=.5*(1+e**2)*math.cos(w) #a1W
    G[2,12]=-.25*e*math.cos(w) #a2W
    G[2,13]=-.5*math.sqrt(1-e**2)*math.sin(w) #b1W
    G[2,14]=.25*e*math.sqrt(1-e**2)*math.sin(w) #b2W
    G[2,:]=G[2,:]*math.sqrt(a/mu)/math.sqrt(1-e^2)

    #Omega
    G[3,9]=-1.5*e*math.sin(w) #a0W
    G[3,10]=.5*(1+e^2)*math.sin(w) #a1W
    G[3,11]=-.25*e*math.sin(w) #a2W
    G[3,12]=.5*math.sqrt(1-e**2)*math.cos(w) #b1W
    G[3,13]=-.25*e*math.sqrt(1-e**2)*math.cos(w) #b2W
    G[3,:]=G[3,:]*math.sqrt(a/mu)*math.csc(i)/math.sqrt(1-e**2)

    #w
    G[4,0]=e*math.sqrt(1-e**2) #a0R
    G[4,1]=-.5*math.sqrt(1-e**2) #a1R
    G[4,7]=.5*(2-e**2) #b1S
    G[4,8]=-.25*e #b2S
    G[4,:]=G[4,:]*math.sqrt(a/mu)/e
    G[4,:]=G[4,:]-math.cos(i)*G[3,:]

    #M
    G[5,0]=-2-e**2 #a0R
    G[5,1]=2*e #a1R
    G[5,3]=-.5*e**2 #a2R
    G[5,:]=G[5,:]*math.sqrt(a/mu)
    G[5,:]=G[5,:]+(1-math.sqrt(1-e**2))*(G[4,:]+G[3,:])+2*math.sqrt(1-e**2)*(math.sin(i/2))**2*G[3,:]-(G[4,:]+G[3,:])

    return G

def findP(A,B,Q,R):

    P = np.zeros(6)
    P[:] = Pvec

    Pdot = -(np.transpose(A)*P+P*A-P*B*(R**-1)*np.transpose(B)*P+Q)

    return Pdot

if __name__=='__main__':
    main()





