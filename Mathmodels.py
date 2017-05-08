""" This file contains both the non-linear 2-body problem and linear LQR"""

import numpy as np
from scipy.interpolate import interp1d

global mu, RE
RE = 6378
mu = 398600 * (60 ** 4)/(RE ** 3)


def ASE(y, t, A, B, R, Q, xT, Pbig, tspan):

    yin = np.reshape(y[0:6], (-1,1))
    x = np.reshape(y[0:6], (-1, 1)) + xT

    if t > tspan[-1]:
        Pvec = Pbig[-1, :]
    else:
        Pvec = interp1d(tspan, Pbig, axis=0)(t)

    P = np.reshape(Pvec, (6, 6))

    nt = np.sqrt(mu/x[0]**3)
    # F = np.array([np.zeros((5, 1)), [nt]])

    u_t = -np.linalg.inv(R).dot(B.T).dot(P).dot(yin)

    dx = A.dot(x) + B.dot(u_t)
    # dJ = np.transpose(yin)*Q*yin + np.transpose(u_t)*R*u_t
    dJ = yin.T.dot(Q).dot(yin) + u_t.T.dot(R).dot(u_t)

    dxl = np.vstack((dx, dJ))

    return dxl.flatten()


def Newt_EOM(y, t, u, tspan):
    """ Differntial 2-body problem with Fourier thrust acceleration"""

    rx, ry, rz = y[0], y[1], y[2]
    vx, vy, vz = y[3], y[4], y[5]

    r = np.array([rx, ry, rz])
    v = np.array([vx, vy, vz])

    r_mag = np.linalg.norm(r)

    # Calculate the Eccentric Anomaly to determine the thrust-acceleration
    r_hat = 1/r_mag * r
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)
    h_hat = h / h_mag
    hr_hat = np.cross(h_hat, r_hat)
    e_vec = 1/mu * np.cross(v, h) - r_hat
    e = np.linalg.norm(e_vec)

    # Check for negative
    if (1-e)/(1+e) < 0:
        print((1-e)/(1+e), '\n')
        print(' e ratio < 0 at \n e: {0}, \t t: {1}'.format(e, t))

    # True Anomaly
    theta = np.arctan2(np.dot(h_hat, np.cross(e_vec, r)), np.dot(e_vec, r))
    # Eccentric Anomaly
    E = 2 * np.arctan2(np.sqrt(abs(1-e)) * np.tan(theta/2), np.sqrt(1+e))

    if E < 0:
        E += 2*np.pi

    if t > tspan[-1]:
        alpha = np.reshape(u[-1, :], (-1))
    else:
        alpha = np.reshape(interp1d(tspan, u, axis=0)(t), (-1))

    # Determine the alpha/beta coefficients for the Thrust Fourier Coefficients

    F_R = alpha[0] + alpha[1]*np.cos(E) + alpha[2]*np.cos(2*E) + alpha[3]*np.sin(E)
    F_S = alpha[4] + alpha[5]*np.cos(E) + alpha[6]*np.cos(2*E) + alpha[7]*np.sin(E) + alpha[8]*np.sin(2*E)
    F_W = alpha[9] + alpha[10]*np.cos(E) + alpha[11]*np.cos(2*E) + alpha[12]*np.sin(E) + alpha[13]*np.sin(2*E)

    thrust = F_R*r_hat + F_S*hr_hat + F_W*h_hat
    c = -mu / (r_mag ** 3)

    dy = np.zeros(7)

    dy[0] = y[3]
    dy[1] = y[4]
    dy[2] = y[5]
    dy[3] = c*y[0]*thrust[0]
    dy[4] = c*y[1]*thrust[1]
    dy[5] = c*y[2]*thrust[2]
    dy[6] = np.sqrt(F_R**2 + F_S**2 + F_W**2)

    return dy
