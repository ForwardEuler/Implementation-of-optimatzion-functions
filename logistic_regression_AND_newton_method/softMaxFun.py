import numpy as np
from scipy.sparse import spdiags, identity
from numpy import matlib as mb
from scipy.linalg import block_diag


def softMaxFun(w, X, Y, arg=None, regFun=None):  
    """
    All vectors are column vectors.
    INPUTS:
        %% Input Parameters:
        X is the (n x d) data matrix.
        w is the (d x C) by 1 weight vector where C is the number of classes (% Technically the total number of classes is C+1, but the degree of freedom is only C).
        Y is the (n x C) label matrix, i.e., Y(i,b) is one if i-th label is class b, and other wise 0.
        arg: output control
        regFun: a function handle of the regulization function
    OUTPUTS:
        f is the function value.
        g is the gradient vector.
        Hv is a function to compute Hessian-vector product.
        H is the explicitly formed Hessian matrix.
    """
    if regFun == None:
        reg_f = 0
        reg_g = 0
        reg_Hv = lambda v: 0
    elif isinstance(regFun, int) is True:
        reg_f = 0
        reg_g = 0
        reg_Hv = lambda v: 0
    else:
        reg_f, reg_g, reg_Hv = regFun(w)
    #X [n x d]
    #Y [n x C]
    global d, C
    n, d = X.shape
    C = int(len(w)/d)
    w = w.reshape(d*C,1) #[d*C x 1]
    W = w.reshape(C,d).T #[d x C]
    XW = np.dot(X,W) #[n x C]
    large_vals = np.amax(XW,axis = 1).reshape(n, 1) #[n,1 ]
    large_vals = np.maximum(0,large_vals) #M(x), [n, 1]
    #XW - M(x)/<Xi,Wc> - M(x), [n x C]
    XW_trick = XW - np.tile(large_vals, (1, C))
    #sum over b to calc alphax, [n x total_C]
    XW_1_trick = np.append(-large_vals, XW_trick,axis = 1)
    #alphax, [n, ]
    sum_exp_trick = np.sum(np.exp(XW_1_trick), axis = 1).reshape(n, 1)
    log_sum_exp_trick = large_vals + np.log(sum_exp_trick)  #[n, 1]
    
    f = np.sum(log_sum_exp_trick)  - np.sum(np.sum(XW*Y,axis=1))  + reg_f
    
    if arg == 'f':        
        return f
    
    inv_sum_exp = 1./sum_exp_trick
    inv_sum_exp = np.tile(inv_sum_exp,(1,np.size(W,axis = 1)))
    S = inv_sum_exp*np.exp(XW_trick) #h(x,w), [n x C] 
    g = np.dot(X.T, S-Y)  #[d x C]
    g = g.T.flatten().reshape(d*C,1) + reg_g#[d*C, ]  
    
    if arg == 'g':
        return g    
    
    if arg == 'fg':
        return f, g
    
    if arg == 'fgHv':    
        Hv = lambda v: hessvec(X, S, n, v) + reg_Hv(v) #write in one function to ensure no array inputs        
        return f, g, Hv    
            
    if arg == 'fgH':
        #S is divided into C parts {1:b}U{c}, [n, ] * C
        S_cell = np.split(S.T,C) 
        SX_cell = np.array([]).reshape(n,0) #empty [n x 0] array
        SX_self_cell = np.array([]).reshape(0,0)
        for column in S_cell:
            c = spdiags(column,0,n,n) #value of the b/c class
            SX_1_cell = np.dot(c.A,X) #WX = W x X,half of W, [n x d]
            #fill results from columns, [n x d*C]
            SX_cell = np.c_[SX_cell, SX_1_cell] 
            SX_cross = np.dot(SX_cell.T,SX_cell) #take square, [d*C x d*C]     
            #X.T x WX        half of W, [d x d]
            SX_1self_cell = np.dot(X.T,SX_1_cell) 
            #put [d x d] in diag, W_cc, [d*C x d*C]  
            SX_self_cell = block_diag(SX_self_cell,SX_1self_cell) 
            H = SX_self_cell - SX_cross #compute W_cc, [d*C x d*C]
        H = H + reg_Hv(identity(d*C))
        return f, g, H

def hessvec(X, S, n, v):
    v = v.reshape(len(v),1)
    V = v.reshape(C, d).T #[d x C]
    A = np.dot(X,V) #[n x C]
    AS = np.sum(A*S, axis=1).reshape(n, 1)
    rep = mb.repmat(AS, 1, C)#A.dot(B)*e*e.T
    XVd1W = A*S - S*rep #[n x C]
    Hv = np.dot(X.T, XVd1W)  #[d x C]
    Hv = Hv.T.flatten().reshape(d*C,1)#[d*C, ] #[d*C, ]
    return Hv
