import numpy as np
import sys
import scipy.linalg as sp
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3


#taking inputs given
if(len(sys.argv)==5):
    Nx=int(sys.argv[1])
    Ny=int(sys.argv[2])
    radius=int(sys.argv[3])  
    Niter=int(sys.argv[4])
    
else:
    Nx=25 # size along x
    Ny=25 # size along y
    radius=8 #radius of central lead
    Niter=1500 #number of iterations to perform
    

#allocating the potential array and initialising it
phi=np.zeros((Nx,Ny))
x=np.linspace(-0.5,0.5,Nx)
y=np.linspace(-0.5,0.5,Ny)
Y,X=np.meshgrid(y,x)
ii=np.where(X**2+Y**2<(0.35)**2)
phi[ii]=1.0


#plot potential
x_c=X[np.where(X**2+Y**2<(0.35)**2)]
y_c=Y[np.where(X**2+Y**2<(0.35)**2)]
plt.figure(num=1)
plt.xlabel("X")
plt.ylabel("Y")
plt.contourf(Y,X[::-1],phi)
plt.plot(x_c,y_c,'ro')
plt.colorbar()
plt.show()


#Functions for the iterations
def update_phi(phi,oldphi):
    phi[1:-1,1:-1]=0.25*(oldphi[1:-1,0:-2]+ oldphi[1:-1,2:]+ oldphi[0:-2,1:-1] + oldphi[2:,1:-1])
    return phi

def boundary(phi):
    phi[:,0]=phi[:,1] # Left Boundary
    phi[:,Nx-1]=phi[:,Nx-2] # Right Boundary
    phi[0,:]=phi[1,:] # Top Boundary
    phi[Ny-1,:]=0
    phi[ii]=1.0
    return phi

err = np.zeros(Niter)
#the iterations
for k in range(Niter):
    oldphi = phi.copy()
    phi = update_phi(phi,oldphi)
    phi = boundary(phi)
    err[k] = np.max(np.abs(phi-oldphi))
    
#plotting Error on semilog
plt.figure(num=2)
plt.title("Error on a semilog plot")
plt.xlabel("No of iterations")
plt.ylabel("Error")
n=np.linspace(0,Niter-1,Niter)
plt.semilogy(n,err)
plt.semilogy(n[::50],err[::50],'ro')
plt.show()

#plotting Error on loglog
nl=n+1
plt.figure(num=3)
plt.title("Error on a loglog plot")
plt.xlabel("No of iterations")
plt.ylabel("Error")
plt.loglog(nl,err)
plt.loglog((nl)[::50],err[::50],'ro')
plt.legend(["real","every 50th value"])
plt.show()

#Function for getting best fit
def fit(y,Niter,lastn=0):
    log_err = np.log(err)[-lastn:]
    X = np.vstack([(np.arange(Niter)+1)[-lastn:],np.ones(log_err.shape)]).T
    log_err = np.reshape(log_err,(1,log_err.shape[0])).T
    return sp.lstsq(X, log_err)[0]

#Function to plot errors with fit
def error_fit(err,Niter,a,a_,b,b_):
    nl= n+1
    plt.figure(num=4)
    plt.title("Best fit for error on a semilog scale")
    plt.xlabel("No of iterations")
    plt.ylabel("Error")
    plt.semilogy(n,err)
    plt.semilogy(n[::100],np.exp(a+b*n)[::100],'ro')
    plt.semilogy(n[::100],np.exp(a_+b_*n)[::100],'go')
    plt.legend(["errors","fit1","fit2"])
    plt.show() 
    
    
    plt.figure(num=5)
    plt.title("Best fit for error on a loglog scale")
    plt.xlabel("No of iterations")
    plt.ylabel("Error")
    plt.loglog(nl,err)
    plt.loglog(nl[::100],np.exp(a+b*nl)[::100],'ro')
    plt.loglog(nl[::100],np.exp(a_+b_*nl)[::100],'go')
    plt.legend(["errors","fit1","fit2"])
    plt.show()
    
    

b,a = fit(err,Niter)
b_,a_ = fit(err,Niter,500)
error_fit(err,Niter,a,a_,b,b_)


#plotting 3d plot of final potential
fig8=plt.figure(num=8)     # open a new figure
ax=p3.Axes3D(fig8) # Axes3D is the means to do a surface plot
plt.title('The 3-D surface plot of the potential')
surf = ax.plot_surface(Y, X, phi.T, rstride=1, cstride=1, cmap=plt.cm.jet)
plt.show()


#plotting 2d contour of final potential
plt.figure(num=7)
plt.title("2D Contour plot of potential")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(x_c,y_c,'ro')
plt.contourf(Y,X[::-1],phi)
plt.colorbar()
plt.show()


#finding Current density
Jx,Jy = (1/2*(phi[1:-1,0:-2]-phi[1:-1,2:]),1/2*(phi[:-2,1:-1]-phi[2:,1:-1]))

#plotting current density
plt.figure(num=9)
plt.title("Vector plot of current flow")
plt.quiver(Y[1:-1,1:-1],-X[1:-1,1:-1],-Jx[:,::-1],-Jy[:,::-1])
plt.plot(x_c,y_c,'ro')
plt.show()


