import numpy as np

def genBlocksTree(N, sz):
    if N == 1:
        Bs = np.array([1])
        upd_shape = np.array([sz])
        
    else:
        Bs = np.array([1])
        while np.max(Bs) < 2: # ensure array isn't all 1s (i.e., trivial shape)
            
            # Generates array with 50/50 split between 1 and 2
            Bs = np.floor(2 * np.random.rand(N,1)) + 1
            #Bs = np.random.randint(1,3,[N,1])
            
            cs_Bs = np.cumsum(Bs, axis=0)
            Bs = Bs[0:np.sum(cs_Bs <= N)]
            if sum(Bs) < N:
                Bs = np.append(Bs, 1)
                
        upd_cs = np.append(0, np.cumsum(Bs))
        upd_shape = np.array([], dtype=int)
        for k in range(0, len(upd_cs)-1):
            app_shape = np.prod(sz[int(upd_cs[k]):int(upd_cs[k+1])])
            upd_shape = np.append(upd_shape, app_shape)
            
        return Bs.flatten(), upd_shape
    
'''
# Use case example
def main():


if __name__ == "__main__":
    main()
    
'''