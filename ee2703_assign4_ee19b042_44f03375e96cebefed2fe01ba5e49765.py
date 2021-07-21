
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt



def exp_periodic(x):
    p=(x//(2*np.pi))
    x=x-(2*np.pi)*p
    return np.exp(x)
def exp(x):
    return np.exp(x)
def coscos(x):
    return np.cos(np.cos(x))




x=np.linspace(-2*np.pi,4*np.pi,300)
exp_x=np.exp(x)
coscos_x=np.cos(np.cos(x))
plt.figure(num=1)
plt.semilogy(x,exp_x,'c',label='Actual Plot')
plt.semilogy(x,exp_periodic(x),color='orange',label='Expected Plot')
plt.legend(loc='upper left')
plt.grid(True)
plt.title('Actual and Expected Plots of $e^x$',fontsize=10)
plt.ylabel(r'$e^{x}\rightarrow$',fontsize=10)
plt.xlabel(r'x$\rightarrow$',fontsize=10)
plt.show()



plt.figure(num=2)
plt.plot(x,coscos_x,'m')
plt.grid(True)
plt.title(' plot of $\cos(\cos(x))$',fontsize=10)
plt.ylabel(r'$\cos(\cos(x))\rightarrow$',fontsize=10)
plt.xlabel(r'x$\rightarrow$',fontsize=10)
plt.show()


func_diction = {'exp(x)': exp,'cos(cos(x))': coscos}
def find_coef(n1,label):
    coef = np.zeros(n1)
    func= func_diction[label]
    u = lambda x,k: func(x)*np.cos(k*x)
    v = lambda x,k: func(x)*np.sin(k*x)
    coef[0]= quad(func,0,2*np.pi)[0]/(2*np.pi)
    for i in range(1,n1,2):
        coef[i] = quad(u,0,2*np.pi,args=((i+1)/2))[0]/np.pi
    for i in range(2,n1,2):
        coef[i] = quad(v,0,2*np.pi,args=(i/2))[0]/np.pi
    return coef

plt.figure(num=3)
e_coef=find_coef(51,'exp(x)')
plt.semilogy(range(51),np.abs(e_coef),'ro')
plt.grid(True)
plt.ylabel(r'exp_coef$\rightarrow$',fontsize=10)
plt.xlabel(r'n$\rightarrow$',fontsize=10)
plt.title('Semilog Plot of Fourier Coefficients of $e^{x}$ ',fontsize=10)
plt.show()

plt.figure(num=4)
plt.loglog(range(51),np.abs(e_coef),'ro')
plt.grid(True)
plt.ylabel(r'exp_coef$\rightarrow$',fontsize=10)
plt.xlabel(r'n$\rightarrow$',fontsize=10)
plt.title('Loglog Plot of Fourier Coefficients of $e^{x}$',fontsize=10)
plt.show()



cos_coef=find_coef(51,'cos(cos(x))')
plt.figure(num=5)
plt.semilogy(range(51),np.abs(cos_coef),'ro')
plt.grid(True)
plt.ylabel(r'coscos_coef$\rightarrow$',fontsize=10)
plt.xlabel(r'n$\rightarrow$',fontsize=10)
plt.title('Semilog Plot of Fourier Coefficients of $\cos(\cos(x))$',fontsize=10)
plt.show()


plt.figure(num=6)
plt.loglog(range(51),np.abs(cos_coef),'ro')
plt.grid(True)
plt.ylabel(r'coscos_coef$\rightarrow$',fontsize=10)
plt.xlabel(r'n$\rightarrow$',fontsize=10)
plt.title('Loglog Plot of Fourier Coefficients of $\cos(\cos(x))$',fontsize=10)
plt.show()


x=np.linspace(0,2*np.pi,401)
x=x[:-1]
b_exp=exp(x)
b_coscos=coscos(x)
A=np.zeros((400,51))
A[:,0]=1
for k in range(1,26):
   A[:,2*k-1]=np.cos(k*x)
   A[:,2*k]=np.sin(k*x)
c1=np.linalg.lstsq(A,b_exp)[0]
c2=np.linalg.lstsq(A,b_coscos)[0]

plt.figure(num=7)
plt.semilogy(range(51),np.abs(c1),'go',label='Least Squares Approach')
plt.semilogy(range(51),np.abs(e_coef),'ro',label='True Value')
plt.grid(True)
plt.xlabel(r'n$\rightarrow$',fontsize=10)
plt.ylabel(r'$Coefficient\rightarrow$',fontsize=10)
plt.title('Semilog Plot of coefficients for $e^{x}$',fontsize=10)
plt.legend(loc='upper right')
plt.show()

plt.figure(num=8)
plt.loglog(range(51),np.abs(c1),'go',label='Least Squares Approach')
plt.loglog(range(51),np.abs(e_coef),'ro',label = 'True Value')
plt.grid(True)
plt.xlabel(r'n$\rightarrow$',fontsize=10)
plt.ylabel(r'$Coefficient\rightarrow$',fontsize=10)
plt.title('Loglog Plot of coefficients of $e^{x}$',fontsize=10)
plt.legend(loc='upper right')
plt.show()

plt.figure(num=9)
plt.semilogy(range(51),np.abs(c2),'go',label='Least Squares Approach')
plt.semilogy(range(51),np.abs(cos_coef),'ro',label='True Value')
plt.grid(True)
plt.xlabel(r'n$\rightarrow$',fontsize=10)
plt.ylabel(r'$Coefficient\rightarrow$',fontsize=10)
plt.title('Semilog Plot of coefficients for $\cos(\cos(x))$',fontsize=10)
plt.legend(loc='upper right')
plt.show()

plt.figure(num=10)
plt.loglog(range(51),np.abs(c2),'go',label='Least Squares Approach')
plt.loglog(range(51),np.abs(cos_coef),'ro',label = 'True Value')
plt.grid(True)
plt.xlabel(r'n$\rightarrow$',fontsize=10)
plt.ylabel(r'$Coefficient\rightarrow$',fontsize=10)
plt.title('Loglog Plot of coefficients of $\cos(\cos(x))$',fontsize=10)
plt.legend(loc='upper right')
plt.show()


dev_exp = abs(e_coef - c1)
dev_cos = abs(cos_coef - c2)
max_dev_exp = np.max(dev_exp)
max_dev_cos = np.max(dev_cos)
print(max_dev_exp)
print(max_dev_cos)
approx_exp = np.matmul(A,c1)
approx_coscos = np.matmul(A,c2)

plt.figure(num=11)
plt.semilogy(x,approx_exp,'go',label="Function Approximation")
plt.semilogy(x,exp(x),'-r',label='True value')
plt.grid(True)
plt.xlabel(r'n$\rightarrow$',fontsize=10)
plt.ylabel(r'$e^{x}\rightarrow$',fontsize=10)
plt.title('Plot of $e^{x}$ and its Fourier series approximation',fontsize=10)
plt.legend(loc='upper right')
plt.show()

plt.figure(num=12)
plt.plot(x,approx_coscos,'go',label="Function Approximation")
plt.plot(x,coscos(x),'-y',label='True value')
plt.grid(True)
plt.xlabel(r'n$\rightarrow$',fontsize=10)
plt.ylabel(r'$\cos(\cos(x))\rightarrow$',fontsize=10)
plt.title('Plot of $cos(cos(x))$ and its Fourier series approximation',fontsize=10)
plt.legend(loc='upper right')
plt.show()
