import scipy
from combat.pycombat import pycombat
import autogenes as ag
import torch
import numpy as np
import qnorm
from scvelo import logging

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3
    
def aggre(ref,meta_ref,var_infor,label,hvg=True):
    ref_mat = np.zeros([ref.shape[0],len(meta_ref[label].values.categories.values)])
    for i in range(len(meta_ref[label].values.categories.values)):
        ct = meta_ref[label].values.categories.values[i]
        re = ref[:,meta_ref[label].values == ct].mean(1)
        ref_mat[:,i] += re
        rownames = var_infor.index.values
        
    if hvg:
        feature = var_infor.index[var_infor.highly_variable_genes.values == "True"].values
        ref_mat = ref_mat[var_infor.highly_variable_genes.values == "True",:]
        rownames = feature
        
    colnames = meta_ref[label].values.categories.values
    ref_mat = pd.DataFrame(ref_mat, columns=colnames, index=rownames)
    return ref_mat

class ENIGMA:
    def __init__(
        self, bulk_data,ref, ref_type,label,
        hvg,device,ncores
    ):
        ## transfer the np.array format data into sparse matrix
        if ref_type != "signature":
            meta_ref = []
            genes = ref.var.index.values
            samples = ref.obs.index.values
            
            if ref_type == "single_cell":
                ## the reference data is from AnnData object
                meta_ref = ref.obs
                ref_m = []
                
            if ref_type == "aggre":
                meta_ref = ref.obs
                var_infor = ref.var
                ref = ref.X.transpose()
                ref_m = aggre(ref,meta_ref,var_infor,label,hvg=hvg)
            
            ### processing bulk datasets
            genes_css = intersection(genes, bulk_data.index)
            bulk_data = bulk_data.loc[genes_css,:]
            if ref_type != "single_cell": ref_m = ref_m.loc[genes_css,:]
            if ref_type == "single_cell": 
                ref = ref[:,genes_css]
                ref = ref.X.transpose()  
        
        if ref_type == "signature":
            meta_ref = []
            samples = []
            ref_m = ref
            genes_css = intersection(ref_m.index, bulk_data.index)
            bulk_data = bulk_data.loc[genes_css,:]
            ref_m = ref_m.loc[genes_css,:]
            
        genes = genes_css 
        self.Bulk = bulk_data
        self.ref_m = ref_m
        self.ref = ref
        self.meta_ref = meta_ref
        self.genes = genes
        self.samples = samples
        self.ref_type = ref_type
        self.remove_batch_effects = []
        self.device = device
        self.ncores = ncores
        
    def remove_batch_effects(self,theta):
        if self.ref_type == "single_cell":
            logging.info("Using S_mode to correct batch effects...")
            self.ref_m,self.Bulk = S_mode_correction(self)
        if self.ref_type != "single_cell":
            logging.info("Using B_mode to correct batch effects...")
            self.Bulk = B_mode_correction(self)    
            
    def get_cell_proportions(self,FSelect = True,optimize=True):
        if FSelect:
            ag.init(self.ref_m.T)
            ag.optimize(ngen=5000,seed=0,nfeatures=400,mode='fixed',offspring_size=100,verbose=False)
            ag.plot(weights=(-1,0))
            index = ag.select(index=0)
            self.fgenes = self.ref_m.index[index].values
        
        #######
        # estimate the abundance
        logging.info("Using robust linear regression model to infer cell type proportions")
        Y = np.array(self.Bulk.loc[self.genes,:])
        X = np.array(self.ref_m.loc[self.genes,:])
        X_scale = np.zeros(X.shape)
        for it in range(X_scale.shape[1]):
            X_scale[:,it] = (X[:,it] - X[:,it].mean())/np.sqrt(np.var(X[:,it])*(X.shape[0]/(X.shape[0]-1)))
        frac_m = np.zeros([Y.shape[1],X.shape[1]])
        for i in range(Y.shape[1]):
            y = Y[:,i]
            if optimize:
                parlist = optimize_para([1.35,0.0001],X,y)
                model = HuberRegressor(epsilon=parlist[0],alpha=parlist[1])
            else:
                model = HuberRegressor()
            model.fit(X,y)
            coef = model.coef_
            coef = np.maximum(coef,0)
            coef = coef / sum(coef)
            frac_m[i,:] = coef
        
        sample_names = self.Bulk.columns
        ct_names = self.ref_m.columns
        frac_m = pd.DataFrame(frac_m, columns=ct_names, index=sample_names)
        
        self.frac_m = frac_m
    
    def run_enigma(self,alpha=0.5,beta=0.1,epsilon=0.001,lr = 1e-1,max_iter=1000,pos=True,normalize=True,norm_method="frac",preprocess="sqrt"):
        
        ###estimate model parameters for ENIGMA
        ##preparing datasets
        X = np.array(self.Bulk)
        theta = np.array(self.frac_m)
        R = np.array(self.ref_m)
        dtype = torch.float
        device = self.device 
        
        if preprocess == "sqrt":
            X = np.sqrt(X)
            
        if preprocess == "log":
            X = np.log(X)
        
        assert preprocess == "log" or preprocess == "sqrt" or preprocess == "raw"
        assert norm_method == "frac" or norm_method == "quantile"
        
        #Convert X into tensor objects
        X = torch.tensor(X, dtype=dtype, device=device)
        #initialization
        logging.info("Initialization")
        P = []
        for i in range(R.shape[1]):
            P.append(torch.tensor(np.array(X), dtype=dtype, device=device))
        P_new = []
        for i in range(R.shape[1]):
            P_new.append(torch.tensor(np.array(X), dtype=dtype, device=device))
        ### create tensor
        stop_flag = False
        __t__ = 1
        #optimizer = torch.optim.Adam(P, lr=lr)
        #schedular = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100, gamma=0.99)
        logging.info("Start Training...")
        for iiter in range(1, max_iter+1):
            
            ## initialize the gradient
            dP = grad(X,theta,P,R,alpha,device)
            for ct_i in range(0,len(P)):
                # initialization
                P[ct_i].grad = torch.zeros_like(P[ct_i])
                P[ct_i].grad.add_(dP[ct_i])
                P_new[ct_i] = proximalpoint(P[ct_i],lr,beta*1e5)
                        
            score = StopMetric(P_new,P)
            logging.info(f'Ratio ranges from:{min(score)} - {max(score)}')
            
            __t__ += 1
            
            if max(score) <= epsilon:
                stop_flag = True
            
            if __t__ >= max_iter:
                stop_flag = True
            
            if not stop_flag:
            #    optimizer.step()
            #    schedular.step()
                P = P_new.copy()
            
            if stop_flag: break
        
        logging.info(f'Converge in {__t__} steps')
        logging.info(f'Perform Normalization...')
        
        for ct_i in range(0,len(P)):
            # initialization
            P[ct_i] = np.maximum(P[ct_i],0)
            
        P_norm = P.copy()
        for i in range(len(P)):
            if preprocess == "log":
                P[i] = np.exp(P[i])-1
            if preprocess == "sqrt":
                P[i] = P[i]**2
                
            if norm_method == "frac":
                props = theta[:,i].reshape(theta.shape[0],1)
                den = P[i] @ props - (len(props)*props.mean()*P[i].mean(1)).reshape(P[i].shape[0],1)
                dev = sum(props**2) - len(props)*props.mean()**2
                GEP = P[i] - den/dev @ props.transpose()
                P_norm[i] = GEP
                
            if norm_method == "quantile":
                df = qnorm.quantile_normalize(P[0].detach().numpy(),ncpus=self.ncores)
                df = torch.tensor(df,dtype = dtype, device = device)
                P_norm[i] = df
        
        self.CSE_normalized = P_norm
        self.CSE = P