from sympy import *
import scipy.signal as sp
import numpy as np
import matplotlib.pyplot as plt


# Sympy function for a lowpass filter.
def lowpass(R1,R2,C1,C2,G,Vi):
	s = symbols('s')
	A = Matrix([[0,0,1,-1/G],
		[-1/(1+s*R2*C2),1,0,0],
		[0,-G,G,1],
		[-1/R1-1/R2-s*C1,1/R2,0,s*C1]])
	b = Matrix([0,0,0,-Vi/R1])
	V = A.inv()*b
	return A,b,V

# Sympy function for a highpass filter.
def highpass(R1,R3,C1,C2,G,Vi):
    s = symbols('s')	
    A = Matrix([[0,-1,0,1/G],
        [s*C2*R3/(s*C2*R3+1),0,-1,0],
        [0,G,-G,1],
        [-s*C2-1/R1-s*C1,0,s*C2,1/R1]])
    b = Matrix([0,0,0,-Vi*s*C1])
    V = A.inv()*b
    return A,b,V

# Function to convert a sympy function into transfer function  	 
def sympyToTrFn(Y):	
    Y = expand(simplify(Y))    
    n,d = fraction(Y)
    n,d = Poly(n,s), Poly(d,s)
    num,den = n.all_coeffs(), d.all_coeffs()
    num,den = [float(f) for f in num], [float(f) for f in den]
    H = sp.lti(num,den)
    return H



# Transfer function for the lowpass filter.
s = symbols('s')
A,b,V = lowpass(10000,10000,1e-9,1e-9,1.586,1)
Vo = V[3]
H = sympyToTrFn(Vo)
ww = np.logspace(0,8,801)
ss = 1j*ww
hf = lambdify(s,Vo,'numpy')
v = hf(ss)

# Step response for the lowpass filter.
A1,b1,V1 = lowpass(10000,10000,1e-9,1e-9,1.586,1/s)
Vo1 = V1[3]
H1 = sympyToTrFn(Vo1)
t,y1 = sp.impulse(H1,None,np.linspace(0,5e-3,10000))

plt.figure(1)
plt.plot(t,y1)
plt.title(r"Step Response for low pass filter")
plt.xlabel(r'$t\rightarrow$')
plt.ylabel(r'$V_o(t)\rightarrow$')
plt.grid(True)
plt.show()


# Output voltage for sum of sinusoids input.
vi = np.sin(2000*np.pi*t) + np.cos(2e6*np.pi*t)
t,y2,svec = sp.lsim(H,vi,t)

plt.figure(2)
plt.plot(t,y2)
plt.title(r"Output voltage for sum of sinusoids")
plt.xlabel(r'$t\rightarrow$')
plt.ylabel(r'$V_o(t)\rightarrow$')
plt.grid(True)
plt.show()

# Transfer function for the highpass filter.
A3,b3,V3 = highpass(10000,10000,1e-9,1e-9,1.586,1)
Vo3 = V3[3]
H3 = sympyToTrFn(Vo3)
hf3 = lambdify(s,Vo3,'numpy')
v3 = hf3(ss)

plt.figure(3)
plt.loglog(ww,abs(v3),lw=2)
plt.title(r"$|H(j\omega)|$ for highpass filter")
plt.xlabel(r'$\omega\rightarrow$')
plt.ylabel(r'$|H(j\omega)|\rightarrow$')
plt.grid(True)
plt.show()

# The output response when the input is a damped sinusoid 
t3 = np.linspace(0,1e-2,1e5)
vi4= np.exp(-1000*t3)*np.cos(2e3*np.pi*t3)
t3,y4,svec = sp.lsim(H3,vi4,t3)

plt.figure(4)
plt.plot(t3,y4)
plt.title(r"Low frequency damped sinusoid response from High Pass filter")
plt.xlabel(r'$t\rightarrow$')
plt.ylabel(r'$V_o(t)\rightarrow$')
plt.grid(True)
plt.show()

# Step response highpass filter.
A5,b5,V5 = highpass(10000,10000,1e-9,1e-9,1.586,1/s)
Vo5 = V5[3]
H5 = sympyToTrFn(Vo5)
t5,y5 = sp.impulse(H5,None,np.linspace(0,5e-3,10000))

plt.figure(5)
plt.plot(t5,y5)
plt.title(r"Step Response for high pass filter")
plt.xlabel(r'$t\rightarrow$')
plt.ylabel(r'$V_o(t)\rightarrow$')
plt.grid(True)
plt.show()	
