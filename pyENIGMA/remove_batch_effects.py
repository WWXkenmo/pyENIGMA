from collections import Counter
from random import choices
import multiprocessing
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from combat.pycombat import pycombat
from sklearn.linear_model import HuberRegressor
import scipy.stats as stats

def AggregateSample(i, frac_m, X, sample_size,celltype_id,sample_id,celltype_label):
    fra = frac_m[i,:]
    sample_vec = np.floor(sample_size * fra)
    
    pseudoGEP = np.zeros([X.shape[0],1])
    ind = sample_vec != 0
    ind = [ind[i] for i in range(len(ind))]
    
    for i in np.array(celltype_label)[ind].tolist():
        label = sample_id[celltype_id == i]
        label = choices(label.tolist(),k=int(sample_vec[[celltype_label[c] == i for c in range(len(celltype_label))]].item()))
        
        for j in label:
            pseudoGEP += X[:,sample_id == j]
        
    return pseudoGEP       


def S_mode_correction(self, n,label, sample_size,ncores):
    GEP = self.ref
    frac = self.meta_ref[label].values.tolist()
    
    val = np.array([Counter(frac)[a] for a in Counter(frac)])
    val = val / sum(val)
    ct_name = [a for a in Counter(frac)]
    
    mean = val
    cov = np.zeros([len(mean),len(mean)])
    for (i,j) in zip(range(len(mean)),range(len(mean))):
        cov[i,j] = mean[i]*2
    
    frac_m = np.random.multivariate_normal(mean, cov, (n,), 'raise')    # nx2
    frac_m = np.maximum(frac_m,0)
    norm_f = frac_m.sum(1)
    frac_m = frac_m[norm_f > 0,:]
    norm_f = norm_f[norm_f>0]

    for i in range(len(frac_m)):
        frac_m[i,:] = frac_m[i,:]/norm_f[i]
        
    ### 
    #Using parallel computing to sample the cells from single cell reference
    num_processes = ncores
    rXTs = np.zeros([self.ref.shape[0],frac_m.shape[0]])
    pool = multiprocessing.pool.ThreadPool(processes=num_processes)
    for i in range(frac_m.shape[0]):
        rXTs[:,[i]] = (pool.apply_async(AggregateSample, args=(
            i,frac_m,self.ref,sample_size,self.meta_ref[label].values,self.samples,ct_name
        )).get())
    pool.close()
    pool.join()
    
    #Using batch effects correction to remove batch effects
    ###convert to pd.Dataframe
    sample_names = []
    for i in range(frac_m.shape[0]):
        name = ["Sample",str(i)]
        name = "_".join(name)
        sample_names.append(name)
      
    rXTs = pd.DataFrame(rXTs, columns=sample_names, index=self.genes)
    assert all(self.genes == self.Bulk.index)
    
    ### merge Bulk
    df_expression = pd.concat([rXTs,self.Bulk],join="inner",axis=1)
    
    ##  generate batch label
    batch = []
    datasets = [rXTs,self.Bulk]
    for j in range(len(datasets)):
        batch.extend([str(j) for _ in range(len(datasets[j].columns))])

    ## correct the batch effects in log space
    log_df_expression = np.log(df_expression+1)
    g = log_df_expression.sum(1)
    logging.info(f"{len(g[g==0])} genes filtered out")
    df = pd.DataFrame({
        "batch": batch})
    log_df_expression = log_df_expression.loc[log_df_expression.index[[i == True for i in g>0]].tolist(),:]
    log_df_corrected = pycombat(log_df_expression,df["batch"])
    df_corrected = np.exp(log_df_corrected[log_df_corrected.columns[[i == "0" for i in batch]]]) - 1    
    
    ## make positive
    df_corrected = np.maximum(df_corrected, 0)

    ## restore in the raw space
    gep_v = np.zeros([frac_m.shape[1],df_corrected.shape[0]])
    for x in range(len(df_corrected)):
        y = df_corrected.iloc[x,:]
        y = np.array(y)
        gep_v[:,x] = nnls(frac_m, y)[0]
        
    gep_v = gep_v.transpose()
    gep_v = pd.DataFrame(gep_v, columns=ct_name, index=df_corrected.index)
    
    return (gep_v,Bulk.loc[gep_v.index,:])
    
def B_mode_correction(self):
    
    model = HuberRegressor()
    Y = np.array(self.Bulk)
    X = np.array(self.ref_m)
    X_scale = stats.zscore(X, axis=1, nan_policy='propagate')
    
    # estimate the abundance
    frac_m = np.zeros([Y.shape[1],X.shape[1]])
    for i in range(Y.shape[1]):
        y = Y[:,i]
        model.fit(X_scale,y)
        coef = model.coef_
        coef = np.maximum(coef,0)
        coef = coef / sum(coef)
        frac_m[i,:] = coef
        
    ### inner product
    pseudoGEP = X @ frac_m.transpose()
    sample_names = self.Bulk.columns + "_pseudo"
    pseudoGEP = pd.DataFrame(pseudoGEP, columns=self.Bulk.columns, index=self.Bulk.index)    
    
    ### remove batch effects
    ### merge Bulk
    df_expression = pd.concat([self.Bulk,pseudoGEP],join="inner",axis=1)
    
    ##  generate batch label
    batch = []
    datasets = [self.Bulk,pseudoGEP]
    for j in range(len(datasets)):
        batch.extend([str(j) for _ in range(len(datasets[j].columns))])

    ## correct the batch effects in log space
    log_df_expression = np.log(df_expression+1)
    g = log_df_expression.sum(1)
    logging.info(f"{len(g[g==0])} genes filtered out")
    df = pd.DataFrame({
        "batch": batch})
    log_df_expression = log_df_expression.loc[log_df_expression.index[[i == True for i in g>0]].tolist(),:]
    log_df_corrected = pycombat(log_df_expression,df["batch"])
    df_corrected = np.exp(log_df_corrected[log_df_corrected.columns[[i == "0" for i in batch]]]) - 1    
    
    ## make positive
    df_corrected = np.maximum(df_corrected, 0)
    
    return df_corrected