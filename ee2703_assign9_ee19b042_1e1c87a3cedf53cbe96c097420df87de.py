from pylab import *
from mpl_toolkits.mplot3d import Axes3D

# Question 1
# Approximate spectrum of sin(sqrt(2)t)
t = linspace(-pi,pi,65); t = t[:-1]
dt = t[1]-t[0]; fmax = 1/dt
y = sin(sqrt(2)*t)
y[0] = 0 
y=fftshift(y)
Y = fftshift(fft(y))/64.0
w = linspace(-pi*fmax,pi*fmax,65); w = w[:-1]

figure(0)
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-4,4])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\sin\left(\sqrt{2}t\right)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),"ro",lw=2)
xlim([-4,4])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
show()  

# Improving the spectrum by windowing
t = linspace(-4*pi,4*pi,257); t = t[:-1]
dt = t[1]-t[0]; fmax = 1/dt
n = arange(256)
wnd = fftshift(0.54+0.46*cos(2*pi*n/256))
y = sin(sqrt(2)*t)*wnd
y[0] = 0 
y=fftshift(y)
Y = fftshift(fft(y))/256.0
w = linspace(-pi*fmax,pi*fmax,257); w = w[:-1]

figure(1)
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-4,4])
ylabel(r"$|Y|$",size=16)
title(r"Improved Spectrum of $\sin\left(\sqrt{2}t\right)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),"ro",lw=2)
xlim([-4,4])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
show() 

#Question 2
# Spectrum of cos^3(0.86t)
y = cos(0.86*t)**3 
y[0]=0
y=fftshift(y)
Y = fftshift(fft(y))/256.0

#plot of Spectrum of cos^3(0.86t) without windowing
figure(2)
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-4,4])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\cos^{3}(0.86t)$ without Hamming window")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),"ro",lw=2)
xlim([-4,4])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
show() 

#plot of Spectrum of cos^3(0.86t) with windowing
y_wn = cos(0.86*t)**3*wnd  
y_wn[0]=0
y_wm=fftshift(y_wn)
Y_wn = fftshift(fft(y_wn))/256.0

figure(3)
subplot(2,1,1)
plot(w,abs(Y_wn),lw=2)
xlim([-4,4])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\cos^{3}(0.86t)$ with Hamming window")
grid(True)
subplot(2,1,2)
plot(w,angle(Y_wn),"ro",lw=2)
xlim([-4,4])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
show() 


#Question 3
w0 = 1.5 
d = 0.5
t = linspace(-pi,pi,129)[:-1]
dt = t[1]-t[0]; fmax = 1/dt
n = arange(128)
wnd = fftshift(0.54+0.46*cos(2*pi*n/128))
y = cos(w0*t + d)*wnd
y[0]=0
y = fftshift(y)
Y = fftshift(fft(y))/128.0
w = linspace(-pi*fmax,pi*fmax,129); w = w[:-1]

figure(4)
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-4,4])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\cos(w_0t+\delta)$ with Hamming window")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),"ro",lw=2)
xlim([-4,4])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
show() 


# w0 -> weighted average of all w>0. Delta -> the phase at w closest to w0.
ii = where(w>=0)
w_cal = sum(abs(Y[ii])**2*w[ii])/sum(abs(Y[ii])**2)
i = abs(w-w_cal).argmin()
delta = angle(Y[i])
print("Calculated value of w0 : ",w_cal)
print("Calculated value of delta: ",delta)


# Question 4 
#Finding w0 and delta for a noisy signal
y = (cos(w0*t + d) + 0.1*randn(128))*wnd
y[0]=0
y=fftshift(y)
Y = fftshift(fft(y))/128.0

figure(5)
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-4,4])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of a noisy $\cos(w_0t+\delta)$ with Hamming window")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),"ro",lw=2)
xlim([-4,4])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
show() 


ii = where(w>=0)
w_cal = sum(abs(Y[ii])**2*w[ii])/sum(abs(Y[ii])**2)
i = abs(w-w_cal).argmin()
delta = angle(Y[i])
print("Calculated value of w0 : ",w_cal)
print("Calculated value of delta: ",delta)


#Question 5 
# Plotting spectrum of a "chirped" signal.
t = linspace(-pi,pi,1025); t = t[:-1]
dt = t[1]-t[0]; fmax = 1/dt
n = arange(1024)
wnd = fftshift(0.54+0.46*cos(2*pi*n/1024))
y = cos(16*t*(1.5 + t/(2*pi)))*wnd
y[0]=0
y = fftshift(y)
Y = fftshift(fft(y))/1024.0
w = linspace(-pi*fmax,pi*fmax,1025); w = w[:-1]

figure(6)
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-100,100])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of chirped function")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),"ro",lw=2)
xlim([-100,100])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
show() 

# Question 6
# Plotting a surface plot with respect to t and w
t = linspace(-pi,pi,1025); t = t[:-1]
t_array = split(t,16)
Y_matrix = zeros((64,16),dtype="complex_")

for i in range(len(t_array)):
	n = arange(64)
	wnd = fftshift(0.54+0.46*cos(2*pi*n/64))
	y = cos(16*t_array[i]*(1.5 + t_array[i]/(2*pi)))*wnd
	y[0]=0
	y = fftshift(y)
	Y = fftshift(fft(y))/64.0
	Y_matrix[:,i] =  Y

t = t[::64]	
w = linspace(-fmax*pi,fmax*pi,64+1); w = w[:-1]
t,w = meshgrid(t,w)

fig1 = figure(7)
ax = fig1.add_subplot(111, projection='3d')
surf=ax.plot_surface(w,t,abs(Y_matrix),cmap='viridis',linewidth=0, antialiased=False)
fig1.colorbar(surf, shrink=0.5)
ax.set_title('surface plot');
ylabel(r"$\omega\rightarrow$")
xlabel(r"$t\rightarrow$")
show()
