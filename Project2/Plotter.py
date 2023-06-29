import numpy as np
import matplotlib.pyplot as plt

data = []

# Open the file and read the content in a list
with open('output.txt', 'r') as filehandle:
    for line in filehandle:
        curr_place = line[:-1]
        data.append(curr_place)
for i in range(len(data)):
    data[i] = eval(data[i])

def Plotter(seq,pos,name): #plots the protein structure in 2D
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
    plt.savefig(name+'.png')
    plt.clf()
    return None
for i in range(len(data)):
    Plotter('00011000',data[i],'images/test'+str(i))
