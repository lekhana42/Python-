import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt

#Question 1 and 2
def func(decay):
    Xnum = np.poly1d([1, decay])
    Xden = np.polymul([1, 0, 2.25], [1, 2*decay, (2.25 + decay**2)])
    Xs = sp.lti(Xnum, Xden)
    t, x = sp.impulse(Xs, None, np.linspace(0, 50, 500))
    return Xs, t, x

# solving for two cases with decay of 0.5 and 0.05
X, t1, x1 = func(0.5)
X, t2, x2 = func(0.05)

# plot of x(t) with decay of 0.5
plt.figure(1)
plt.plot(t1, x1,'r')
plt.title(r"$x(t)$ with decay=0.5")
plt.xlabel(r"$t \to $")
plt.ylabel(r"$x(t) \to $")
plt.show()

#plot of x(t) with decay 0.05
plt.figure(2)
plt.plot(t2, x2,'r')
plt.title(r"$x(t)$ with decay=0.05")
plt.xlabel(r"$t \to $")
plt.ylabel(r"$x(t) \to $")
plt.show()

#Question 3
H = sp.lti([1],[1,0,2.25])
freq=np.linspace(1.4,1.6,5)
for w in freq:
	t = np.linspace(0,50,500)
	f = np.cos(w*t)*np.exp(-0.05*t)
	t,x,svec = sp.lsim(H,f,t)

# The plot of x(t) for various frequencies vs time.
	plt.figure(3)
	plt.plot(t,x,label='w = ' + str(w))
	plt.title("x(t) for different frequencies")
	plt.xlabel(r'$t\rightarrow$')
	plt.ylabel(r'$x(t)\rightarrow$')
	plt.legend(loc = 'upper left')  
	plt.show()

#Question 4
t4 = np.linspace(0,20,500)
X4 = sp.lti([1,0,2],[1,0,3,0])
Y4 = sp.lti([2],[1,0,3,0])	
t4,x4 = sp.impulse(X4,None,t4)
t4,y4 = sp.impulse(Y4,None,t4)

#plot of x(t)
plt.figure(4)   
plt.plot(t4, x4,'r')
plt.title(r"Time evolution of $x(t)$ for Coupled spring system")
plt.xlabel(r"$t \to $")
plt.ylabel(r"$x(t) \to $")
plt.show()

#plot of y(t)
plt.figure(5)   
plt.plot(t4, y4,'r')
plt.title(r"Time evolution of $y(t)$ for Coupled spring system ")
plt.xlabel(r"$t \to $")
plt.ylabel(r"$y(t) \to $")
plt.show()


def RLC_transfunc(R, C, L):
    Hnum = np.poly1d([1])
    Hden = np.poly1d([L*C, R*C, 1])
    Hs = sp.lti(Hnum, Hden)
    w, mag, phi = Hs.bode()
    return w, mag, phi, Hs

w, mag, phi, H = RLC_transfunc(100, 1e-6, 1e-6)

# plot Magnitude Response
plt.figure(6)
plt.semilogx(w, mag)
plt.title(r" Magnitude Response of $H(jw)$ of Series RLC network")
plt.xlabel(r"$ w \to $")
plt.ylabel(r"$ 20\log|H(jw)|  \to $")
plt.show()

# Plot of phase response
plt.figure(7)
plt.semilogx(w, phi, 'r', label="$Phase Response$")
plt.title(r"Phase response of the $H(jw)$ of Series RLC network")
plt.xlabel(r"$  w \to $")
plt.ylabel(r"$ \angle H(j\omega)$ $\to $")
plt.show()

#Question 6
t = np.arange(0, 90*10**-3,10**-6)
vi = np.cos(t*10**3)-np.cos(t*10**6)
t, vo, svec = sp.lsim(H, vi, t)

# Plot of Vo(t) for large time 
plt.figure(8)
plt.plot(t, vo, 'r')
plt.title(r"Output voltage $v_0(t)$ at Steady State")
plt.xlabel(r"$ t \to $")
plt.ylabel(r"$ y(t) \to $")
plt.show()

# Plot of Vo(t) for 0<t<30usec
plt.figure(9)
plt.plot(t, vo, 'r')
plt.title(r"Output voltage $v_0(t)$ for $0<t<30\mu sec$")
plt.xlim(0, 30*(10**(-6)))
plt.ylim(-1e-5, 0.3)
plt.xlabel(r"$ t \to $")
plt.ylabel(r"$ v_0(t) \to $")
plt.show()
