#perform ASE targeting as an LQ optimal control algorithm

import numpy
from scipy import integrate
import math
from math import pi


def main():
    # Constants
    earth_radius = 6378
    mu = 398600 * 60 ** 4 / earth_radius ** 3
    # Initial Orbit State
    a0 = 6678 / RE  # km
    e0 = 0.67
    i0 = 20 * math.pi / 180  # rad
    Omega0 = 20 * math.pi / 180  # rad
    w0 = 20 * pi / 180  # rad
    M0 = 20 * pi / 180  # rad
    n0 = math.sqrt(mu / a0 ** 3)
    x0 = numpy.array([[a0], [e0], [i0], [Omega0], [w0], [M0]])
    # transfer time
    t0 = 0
    ttarg = 2 * pi * math.sqrt(a0 ** 3 / mu) * 20
    dt = ttarg / 500
    tspan = numpy.array([range(t0, ttarg, dt)])
    tspan_bk = numpy.array([range(ttarg, dt, 0)])
    # convert oe to rv
    oe_to_rv(oe, t)



def oe_to_rv(oe,t):
    a=oe[0]
    e=oe[1]
    i=oe[2]
    Omega=oe[4]
    w=oe[5]
    M=oe[6]
    #%Determine orbit type
    if a<0 or e<0 or e>1 or math.fabs(i)>2*math.pi or math.fabs(Omega)>2*math.pi or math.fabs(w)>2*math.pi: #problem
        print(a)
        print(e)
        print(i)
        print(Omega)
        print(w)
        print('Invalid orbital element(s)')
    xhat=numpy.array([1, 0, 0])
    yhat=numpy.array([0, 1, 0])
    zhat=numpy.array([0, 0, 1])

    nu=my_kepler(oe, t, mu)
    nhat=math.cos(Omega)*xhat+math.sin(Omega)*yhat
    # hhat=sin(i)*sin(Omega)*xhat-sin(i)*cos(Omega)*yhat+cos(i)*zhat
    rhatT=-math.cos(i)*math.sin(Omega)*xhat+math.cos(i)*math.cos(Omega)*yhat+math.sin(i)*zhat
    rmag=a*(1-e^2)/(1+e*math.cos(nu))
    vmag=math.sqrt(mu/rmag*(2-rmag/a))
    gamma=math.atan2(e*math.sin(nu),1+e*math.cos(nu))
    u=w+nu
    rhat=math.cos(u)*nhat+math.sin(u)*rhatT
    vhat=sin(gamma-u)*nhat+cos(gamma-u)*rhatT
    r=rmag*rhat
    v=vmag*vhat


def kepler(oe,t,mu):
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
                dx = -f/(fp+dx*fpp/2+dx^2*fppp/6)
                x = x+dx
    if count == 10: #check that Newton's method converges
          nu = 'undefined'
    #    else %test that computations were correct
    #       time=(E-e*sin(E))/sqrt(mu/a^3)+Tau
    else:
        nu = 0
        E = 0
    #time=(E-e*math.sin(E))/math.sqrt(mu/a^3)+Tau

















