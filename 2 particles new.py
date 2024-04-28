#LIBRARIES

import numpy as np
import math as mt
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import itertools

#VALUE DEFINITION

s=5         #dimension of lattice
n=2         #number of particles
a=2         #initial position
mu=1        #value of mu
tau=1       #value of tau
U=1         #value of U

t=0         #initial time
t_f=10      #final time
nt=500      #number of iterations

d=mt.comb(s+n-1,n) #dimension of the Hilbert space

#OPERATOR MATRICES

#Hilbert space basis

def generate_vector_basis(dimension,particles):
    basis=[]
    for combo in itertools.combinations_with_replacement(range(dimension),particles):
        vector=[0]*dimension
        for index in combo:
            vector[index]+=1
        basis.append(vector)
    return basis

basis=np.array(generate_vector_basis(s,n))
file_path="basis.txt"
np.savetxt(file_path,basis,fmt='%.0f')

#Number operator matrices

n_op={}
for i in range(s):
    name=f"n_{i+1}"
    matrix=np.zeros((d,d))
    for j in range(d):
        vector=basis[j]
        matrix[j,j]=np.sqrt(vector[i])
    n_op[name]=matrix

file_path="n_op.txt"
np.savetxt(file_path,n_op["n_4"],fmt='%.4f')

#Hamiltonian operator matrix

H_int=np.zeros((d,d))                       #interaction terms
for vector in basis:
    index=np.where((basis==vector).all(axis=1))
    for value in vector:
        if value > 1:
            H_int[index[0],index[0]]+=value*(value-1)

H_hop=np.zeros((d,d))                       #hopping terms
H_hop[np.arange(d-1),np.arange(1,d)]=1
H_hop[np.arange(1,d),np.arange(d-1)]=1

H=tau*H_hop+U/2*H_int                       #Hamiltonian
file_path="H.txt"
np.savetxt(file_path,H,fmt='%.0f')

#Data storage matrices

psi_t=np.empty((nt+1,d),dtype=complex)
time=np.empty((nt+1,1))
prob=np.empty((nt+1,s))

#INITIAL STATE

psi_0_old=np.zeros(s)
psi_0_old[3]=1
psi_0_old[1]=1

psi_0=np.zeros((d,1))
index=np.where((basis==psi_0_old).all(axis=1))
psi_0[index[0]]=1

#COMPUTATION

#Time evolution of \Psi

for k in range(nt+1):
    psi_tt=(np.dot(expm(-1j*t*H),psi_0))
    psi_t[k,:d]=np.transpose(psi_tt)
    time[k,0]=t
    t=(k+1)*t_f/nt

#Expected value <n>:

    for j in range(s):
        prob[k,j]=np.abs(np.vdot(np.dot(np.transpose(np.conj(psi_tt)),n_op[f"n_{j+1}"]),psi_tt))

#Data storage

data=np.concatenate((prob,time),axis=1)
file_path="data.txt"
np.savetxt(file_path,data,fmt='%.4f')

#PLOT

colormap=plt.get_cmap()             #type of map
norm=plt.Normalize(0,1)             #normalize probabilities
fig,ax=plt.subplots()

for i in range(data.shape[1]-1):    #iterate over "pictures" in time
    x=data[:,i]                     #probability values
    y=data[:,s]                     #time values

    colors=colormap(norm(x))        #convert probabilities to colors based on the colormap

    sc=ax.scatter(np.repeat(i+1,len(x)),y,s=100,c=x,cmap='binary',label=f'Psi{i+1}',marker='_')

ax.set_xlabel('Positions')
ax.set_ylabel('Time')
ax.set_title('Probabilities vs Time')
ax.set_xticks(np.arange(data.shape[1]-1)+1)
ax.set_xticklabels([f'Psi {i+1}' for i in range(data.shape[1] - 1)]) #name the columns

#cbar = plt.colorbar(sc, ax=ax, label='Probability')

plt.grid(True)
plt.show()
