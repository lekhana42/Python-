from pylab import *

# Spectrum of sin(5t)
t1 = linspace(0,2*pi,129); t1 = t1[:-1]
y1 = sin(5*t1)
Y1 = fftshift(fft(y1))/128
w1 = linspace(-64,63,128)

figure(1)
subplot(2,1,1)
plot(w1,abs(Y1))
xlim([-10,10])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\sin(5t)$")
grid(True)
subplot(2,1,2)
plot(w1,angle(Y1),'ro')
ii=where(abs(Y1)>1e-3)
plot(w1[ii],angle(Y1[ii]),'go')
xlim([-10,10])
ylabel(r"Phase of $Y$")
xlabel(r"$k$")
grid(True)
show()

# Spectrum of (1 + 0.1cos(t))cos(10t)
t2 = linspace(-4*pi,4*pi,513); t2 = t2[:-1]
y2 = (1 + 0.1*cos(t2))*cos(10*t2)
Y2 = fftshift(fft(y2))/512
w2 = linspace(-64,64,513); w2 = w2[:-1]

figure(2)
subplot(2,1,1)
plot(w2,abs(Y2))
xlim([-15,15])
ylabel(r"$|Y|$")
title(r"Spectrum of $\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)$")
grid(True)
subplot(2,1,2)
plot(w2,angle(Y2),'ro')
ii=where(abs(Y2)>1e-3)
plot(w2[ii],angle(Y2[ii]),'go')
xlim([-15,15])
ylabel(r"Phase of $Y$")
xlabel(r"$\omega$")
grid(True)
show()

# Spectrum of sin^3(t)
t3 = linspace(-4*pi,4*pi,513); t3 = t3[:-1]
y3 = (3*sin(t3) - sin(3*t3))/4
Y3 = fftshift(fft(y3))/512
w3 = linspace(-64,64,513); w3 = w3[:-1]

figure(3)
subplot(2,1,1)
plot(w3,abs(Y3))
title(r"Spectrum of $sin^{3}(t)$")
ylabel(r"$|Y(\omega)|\rightarrow$")
xlabel(r"$\omega\rightarrow$")
xlim([-15,15])
grid(True)
subplot(2,1,2)
plot(w3,angle(Y3),'ro')
ii=where(abs(Y3)>1e-3)
plot(w3[ii],angle(Y3[ii]),'go')
xlim([-15,15])
ylabel(r"Phase of $Y$")
xlabel(r"$\omega$")
grid(True)
show()


# Spectrum of cos^3(t)
t4 = linspace(-4*pi,4*pi,513); t4 = t4[:-1]
y4 = (3*cos(t4) + cos(3*t4))/4
Y4 = fftshift(fft(y4))/512
w4 = linspace(-64,64,513); w4 = w4[:-1]

figure(4)
subplot(2,1,1)
plot(w4,abs(Y4))
title(r"Spectrum of $cos^{3}(t)$")
ylabel(r"$|Y(\omega)|\rightarrow$")
xlabel(r"$\omega\rightarrow$")
xlim([-15,15])
grid(True)
subplot(2,1,2)
plot(w4,angle(Y4),'ro')
ii=where(abs(Y4)>1e-3)
plot(w4[ii],angle(Y4[ii]),'go')
xlim([-15,15])
ylabel(r"Phase of $Y$")
xlabel(r"$\omega$")
grid(True)
show()

# Spectrum of cos(20t + 5cos(t))
t5 = linspace(-4*pi,4*pi,513); t5 = t5[:-1]
y5 = cos(20*t5 + 5*cos(t5))
Y5 = fftshift(fft(y5))/512
w5 = linspace(-64,64,513); w5 = w5[:-1]


figure(5)
subplot(2,1,1)
plot(w5,abs(Y5))
title(r"Spectrum of $cos(20t + 5cos(t))$")
ylabel(r"$|Y(\omega)|\rightarrow$")
xlabel(r"$\omega\rightarrow$")
xlim([-40,40])
grid(True)
subplot(2,1,2)
ii = where(abs(Y5)>=1e-3)
plot(w5[ii],angle(Y5[ii]),'go')
ylabel(r"Phase of $Y$")
xlabel(r"$\omega\rightarrow$")
xlim([-40,40])
grid(True)
show()

#To find out the most accurate DFT of a Gaussian
T = 2*pi
N = 128
tolerance = 1e-6

#loop to find out when the error is less than tolerance
while True:

	t = linspace(-T/2,T/2,N+1)[:-1]
	w = N/T * linspace(-pi,pi,N+1)[:-1] 
	y = exp(-0.5*t**2)

	Y = fftshift(fft(y))*T/(2*pi*N)
	Y_actual = (1/sqrt(2*pi))*exp(-0.5*w**2)
	Y6=abs(Y)
	error = mean(abs(Y6-Y_actual))

	if error < tolerance:
		break
	
	T = T*2
	N = N*2


# Accurate DFT of the Gaussian. 
figure(6)
subplot(2,1,1)
plot(w,abs(Y))
title(r"Estimated spectrum of a Gaussian function")
ylabel(r"$|Y(\omega)|\rightarrow$")
xlabel(r"$\omega\rightarrow$")
grid(True)
xlim([-10,10])
subplot(2,1,2)
plot(w,angle(Y),'ro')
ylabel(r"Phase of $Y$")
xlabel(r"$\omega\rightarrow$")
xlim([-10,10])
grid(True)
show()

#Real spectrum of Gaussian
figure(7)
subplot(2,1,1)
plot(w,abs(Y_actual))
title(r"Real Spectrum of a Gaussian function")
ylabel(r"$|Y(\omega)|\rightarrow$")
xlabel(r"$\omega\rightarrow$")
grid(True)
xlim([-10,10])
subplot(2,1,2)
plot(w,angle(Y_actual),'ro')
ylabel(r"Phase of $Y$")
xlabel(r"$\omega\rightarrow$")
xlim([-10,10])
grid(True)
show()


