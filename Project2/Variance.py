import numpy as np

n=20
L=9
pos=[[i,0] for i in range(8)]
Mx2 = np.full((L,L),np.nan)

for l in np.arange(3,L+1,1):
    print(l/(L+1))
    for hs in range(l+1):
        seq = np.ones((n,l))
        for i in range(n):
            Hposns = np.random.choice(np.arange(0,l,1),(hs),replace=False)
            for j in Hposns:
                seq[i][j] = 0
        Var = []
        for i in range(n):
            seqStr = ''.join(str(int(k)) for k in seq[i])
            finalPosns = metropolis(seqStr,pos,'2')[-100:-1]
            
            Var.append(np.var([h(seqStr,pos_) for pos_ in finalPosns]))
        Mx2[l][hs] = np.mean(Var)
        
with open('Mx2.npy', 'wb') as f:
    np.save(f, Mx2)