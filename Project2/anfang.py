from __future__ import with_statement
import numpy as np
import matplotlib.pyplot as plt
import csv
L=8
N=2**L
liste_seq=[] #list of all possible sequences with length L
#define states polar as P=1 hydrophobic as H=0
for i in range(N):
    new=bin(i)
    strnew=str(new)[2:]
    while len(strnew)<8:
        strnew='0'+strnew
    liste_seq.append(strnew)

def Plotter(seq,pos,name='0'): #plots the protein structure in 2D
    L=len(seq)
    H=False
    P=False
    for i in range(L):
        if seq[i]=='0':
            if H==False:
                plt.scatter(pos[i][0],pos[i][1],80, color='blue',facecolors='none',label='H')
                H=True
            else:
                plt.scatter(pos[i][0],pos[i][1],80, color='blue',facecolors='none')
        else:
            if P==False:
                plt.scatter(pos[i][0],pos[i][1],80, color='blue',label='P')
                P=True
            else:
                plt.scatter(pos[i][0],pos[i][1],80, color='blue')
        if i<L-1:
            plt.plot([pos[i][0],pos[i+1][0]],[pos[i][1],pos[i+1][1]], color='blue')
    plt.legend(loc = "upper left")
    if name!='0':
        plt.savefig(name+'.png')
    plt.show()
    return None

seq=liste_seq[int(N/3)]
pos=[[i,0] for i in range(12)] #strating postion
#pos[0]=[1,-1]
#print(pos)

# Determine possbile follow-up structures with move set 1
# 4 different sequences based on ends switching
# 4 different states for the end of a sequence
# 1 move per corner
def Adjacent_structures(seq,pos):
    Struc=[pos]
    L=len(seq)
    a=[[pos[1][0]-1,pos[1][1]],[pos[1][0]+1,pos[1][1]],[pos[1][0],pos[1][1]-1],[pos[1][0],pos[1][1]+1]]
    counter=0
    for poss in a:
        if poss not in pos:
            counter+=1
    for possibility in a:
        jumpsize=np.abs(np.abs(possibility[0])-np.abs(pos[0][0]))+np.abs(np.abs(possibility[1])-np.abs(pos[0][1]))
        if possibility not in pos and (jumpsize<2 or counter>1):
            Struc2=pos.copy()
            Struc2[0]=possibility
            if Struc2 not in Struc:
                Struc.append(Struc2)
    a=[[pos[-2][0]-1,pos[-2][1]],[pos[-2][0]+1,pos[-2][1]],[pos[-2][0],pos[-2][1]-1],[pos[-2][0],pos[-2][1]+1]]
    for poss in a:
        if poss not in pos:
            counter+=1
    for possibility in a:
        jumpsize=np.abs(np.abs(possibility[0])-np.abs(pos[-1][0]))+np.abs(np.abs(possibility[1])-np.abs(pos[-1][1]))
        if possibility not in pos and (jumpsize<2 or counter>1):
            Struc2=pos.copy()
            Struc2[-1]=possibility
            if Struc2 not in Struc:
                Struc.append(Struc2)
    corners=[]
    for i in range(L-2):
        if (np.abs(pos[i][0]-pos[i+2][0])+np.abs(pos[i][0]-pos[i+2][0]))== 2:
            corners.append(i+1)
            cor_move=[[pos[i][0],pos[i+2][1]],[pos[i+2][0],pos[i][1]]]
            for move in cor_move:
                if move not in pos:
                    Struc2=pos.copy()
                    Struc2[i+1]=move
                    if Struc2 not in Struc:
                        Struc.append(Struc2)
    return Struc

def Prob(eps,pos_old,pos_new,seq,all):
    h_old=h(seq,pos_old)
    h_new=h(seq,pos_new)
    return min([1,np.exp(-eps*(h_new-h_old))])*1/all

def h(seq,pos):
    contacts=[]
    number=0
    L=len(seq)
    for i in range(L-2):
        for j in range(i+2,L):
            if np.abs(np.abs(pos[i][0])-np.abs(pos[j][0]))+np.abs(np.abs(pos[i][1])-np.abs(pos[j][1]))==1:
                contacts.append([i,j])
    for pair in contacts:
        if seq[pair[0]] == '0' and seq[pair[1]] == '0':
            number+=1
    return number

def metropolis(seq,start):
    pos=start
    positions=[]
    positions.append(pos)
    eps=-4
    for j in range(1000):
        adj_struct=Adjacent_structures(seq,pos)
        areas=[]
        last=0
        for i in range(1,len(adj_struct)):
            if len(areas)>0:
                last=areas[-1]
            areas.append(Prob(eps,pos,adj_struct[i],seq,len(adj_struct))+last)
        u=np.random.uniform()
        if u<=areas[-1]:
            for i in range(len(areas)):
                if u<areas[i]:
                    area=i
                    break
            pos=adj_struct[area+1]
            positions.append(pos)
        else:
            positions.append(pos)
        #print(str(j))
        #Plotter(seq,pos,'test'+str(j))
    return positions
'''
M=Adjacent_structures(seq,pos)
for struct in M:
    Plotter(seq,struct)
'''
seq='00011000'
pos=[[i,0] for i in range(8)]
result=metropolis(seq,pos)

with open('output.txt', 'w') as f:
    for _list in result:
            f.write(str(_list) + '\n')
