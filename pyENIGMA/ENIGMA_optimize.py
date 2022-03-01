## import functions

def ident(vec):
    m = np.zeros([len(vec),len(vec)])
    for (i,j) in zip(range(len(vec)),range(len(vec))):
        m[i,j] = vec[i]
    return m

def diag(vec):
    m = np.zeros([len(vec),len(vec)])
    for (i,j) in zip(range(len(vec)),range(len(vec))):
        m[i,j] = vec[i]
    return m

def grad(X,theta,P_old, R, alpha,device):
    dP1 = []
    for i in range(R.shape[1]):
        dP1.append(torch.zeros([X.shape[0],X.shape[1]], dtype=torch.float, device=device,requires_grad=False))
    
    dP2 = dP1.copy()
    
    for cell_type_index in range(0,len(dP1)):
        R_m = R[:,cell_type_index]
        
        cell_type_seq = np.array(range(len(P_old)))
        cell_type_seq = cell_type_seq[cell_type_seq!=cell_type_index]
        
        X_summary = torch.zeros([X.shape[0],X.shape[1]], dtype=torch.float, device=device,requires_grad=False)
        for i in cell_type_seq:
            X_summary += P_old[i] @ diag(theta[:,i])
        X_summary = X - X_summary
        
        dP1[cell_type_index] = 2*(P_old[cell_type_index] @ diag(theta[:,cell_type_index]) - X_summary) @ diag(theta[:,cell_type_index])
        dP1[cell_type_index] = torch.tensor(dP1[cell_type_index],dtype=torch.float, device=device,requires_grad=False)
        dP2_m1 = np.matrix(2*(P_old[cell_type_index].mean(1) - R[:,cell_type_index])).transpose()
        dP2_m2 = np.repeat(1/theta.shape[0],theta.shape[0]).reshape(1,theta.shape[0])
        dP2[cell_type_index] = dP2_m1 @ dP2_m2
        dP2[cell_type_index] = torch.tensor(dP2[cell_type_index],dtype=torch.float, device=device,requires_grad=False)
    
    dims = dP1[0].shape[0] * dP1[0].shape[1]
    norm_factor1 = torch.tensor([0],dtype=torch.float, device=device,requires_grad=False)
    for i in range(len(dP1)): norm_factor1 += sum(dP1[i].reshape(1,dims) * dP1[i].reshape(1,dims)).sum()
    norm_factor1 = np.sqrt(norm_factor1)
    
    norm_factor2 = torch.tensor([0],dtype=torch.float, device=device,requires_grad=False)
    for i in range(len(dP1)): norm_factor2 += sum(dP2[i].reshape(1,dims) * dP2[i].reshape(1,dims)).sum()
    norm_factor2 = np.sqrt(norm_factor2)

    for i in range(len(dP1)): dP1[i] = dP1[i]*1e5/norm_factor1.item()
    for i in range(len(dP2)): dP2[i] = dP2[i]*1e5/norm_factor2.item()
    
    dP = dP1.copy()
    for i in range(len(dP1)): dP[i] = alpha*dP1[i] + (1-alpha)*dP2[i]
    
    return dP
    

def proximalpoint(P,tao_k,beta,device="cpu"):
    V = P - tao_k*P.grad
    V = V.t()
    V = V.detach().numpy()
    V = torch.tensor(V,dtype=torch.float,device = device,requires_grad=False)
    P_hat = squash(V,tao_k*beta)
    P_hat = P_hat.t()
    #grad_new = (P - P_hat)/tao_k
    
    return P_hat

def StopMetric(P_new,P):
    score = []
    for i in range(len(P)):
        D = (P_new[i] - P[i])**2
        score.append(D.sum().item())
    
    return score

def squash(V,beta):
    M = V.clone()
    ## squash: calculate the optimal solution of the formula: X=argmin{ (||X-V||_F)^2 + beta*||X||_2_max }
    n = []
    for i in range(V.shape[0]):
        n.append(np.sqrt(sum(V[i,:]*V[i,:])))
    n = np.array(n)
    pi = torch.sort(torch.tensor(n),descending=True).indices
    s = []
    for i in range(len(pi)):
        i = i+1
        s.append(n[pi[0:i]].sum())
    s = np.array(s)
    dev = (np.array(range(len(s)))+1)+beta
    s_n = s/dev
    q = (np.array(range(len(n)))+1)[n[pi]>=s_n]
    q = q.max()
    tao = s[q-1]/(q+beta)
    for i in range(q):
        M[pi[i],:] = tao*M[pi[i],]/np.sqrt(sum(M[pi[i],:]*M[pi[i],:]))      
    return M
    