import numpy as np
import sys
import matplotlib.pyplot as plt

#taking inputs given
if(len(sys.argv)==6):
    n=(int)(sys.argv[1])
    M=(int)(sys.argv[2])
    nk=(int)(sys.argv[3])
    u0=float(sys.argv[4])
    p=float(sys.argv[5])
else:
    n=100 # spatial grid size.
    M=5 # number of electrons injected per turn.
    nk=500 # number of turns to simulate.
    u0=5 # threshold velocity.
    p=0.25 # probability that ionization will occur


#vectors for electron informaton
xx=np.zeros(n*M)	
u=np.zeros(n*M)	
dx=np.zeros(n*M)	
I=[]	
X=[]	
V=[]	


for k in range(nk):
   
    #electrons whose position greater than 0
    ii=np.where(xx>0) 
    #computing displacement and electron position and velocity advancement
    dx[ii]=u[ii]+0.5         	
    xx[ii]=xx[ii]+dx[ii]     
    u[ii]=u[ii]+1            
    
    #electrons that hit the anode
    i_anode=np.where(xx>=n)
    #setting position,velocity,displacement zero
    xx[i_anode]=0
    u[i_anode]=0
    dx[i_anode]=0
    
    #finding electrons whose velocity is greater than threshold
    kk=np.where(u>=u0)[0]
    #ionised electrons
    ll=np.where(np.random.rand(len(kk))<=p)[0]
    kl=kk[ll]
    #setting velocity zero
    u[kl]=0
    
    xx[kl]=xx[kl]-(dx[kl]*np.random.rand())
    
    I.extend(xx[kl].tolist())
    
    #for injected electrons
    Msig=2
    m=int((np.random.rand()*Msig)+M)
    iu=np.where(xx==0)[0]	
    xx[iu[:m]]=1
    
    ii=np.where(xx>0)
    X.extend(xx[ii].tolist())
    V.extend(u[ii].tolist())
    
    
    
    
#electron density plot
plt.figure(num=0)
plt.hist(X,bins=n)
plt.xlabel('x')
plt.ylabel('X')
plt.title('Electron Density')
plt.show()

#intensity plot
plt.figure(num=1)
count,bins,l=plt.hist(I,bins=n)
plt.xlabel('x')
plt.ylabel('I')
plt.title('Light Intensity')
plt.show()


#electron phase plot
plt.figure(num=2)
plt.scatter(X,V,marker='x')
plt.xlabel('X')
plt.ylabel('V')
plt.title('Electron phase space')
plt.show()

xpos=0.5*(bins[0:-1]+bins[1:])
print("Intensity data:")
print("xpos\t\t\tcount")
for i in range(n):
    print(xpos[i],"\t",count[i])


