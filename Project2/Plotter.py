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

def connection(seq, pos):
    
    # loop over connectios
    for i in range(len(seq)):
        # is H 
        if seq[i] == "0":
            r = pos[i]
            for k in range(2):
                if [r[0]+(2*k-1), r[1]] in pos:
                    # brute force search
                    for j in range(i+2, len(seq)):
                        if pos[j][0] == r[0]+(2*k-1) and pos[j][1] == r[1] and seq[j] == "0":
                            plt.plot([r[0], pos[j][0]], [r[1], pos[j][1]], "r-", alpha=0.5)

                if [r[0], r[1]+(2*k-1)] in pos:
                    # brute force search
                    for j in range(i+2, len(seq)):
                        if pos[j][0] == r[0] and pos[j][1] == r[1]+(2*k-1) and seq[j] == "0":
                            plt.plot([r[0], pos[j][0]], [r[1], pos[j][1]], "r-", alpha=0.5)

    return None



def Plotter(seq,pos,name): #plots the protein structure in 2D
    L=len(seq)
    H=False
    P=False
    for i in range(L):
        if seq[i]=='0':
            if H==False:
                plt.scatter(pos[i][0],pos[i][1],80, color='blue',facecolors='none',label='H', alpha=1)
                H=True
            else:
                plt.scatter(pos[i][0],pos[i][1],80, color='blue',facecolors='none', alpha=1)
        else:
            if P==False:
                plt.scatter(pos[i][0],pos[i][1],80, color='blue',label='P', alpha=1)
                P=True
            else:
                plt.scatter(pos[i][0],pos[i][1],80, color='blue', alpha=1)
        if i<L-1:
            plt.plot([pos[i][0],pos[i+1][0]],[pos[i][1],pos[i+1][1]], color='blue', alpha=1)
    connection(seq, pos)
    plt.axis([-1, 9, -5, 5])
    plt.grid()
    plt.legend(loc = "upper left")
    plt.savefig(name+'.png')
    plt.clf()
    return None
for i in range(len(data)):
    Plotter('00011000',data[i],'images/test'+str(i))
