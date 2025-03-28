from collections import OrderedDict
import re
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm.notebook import tqdm
from tasks import PolynomialRegression, ChebyshevPolynomialRegression
import numpy as np
from eval import get_run_metrics, read_run_dir, get_model_from_run
from plot_utils import basic_plot, collect_results, relevant_model_names
from samplers import get_data_sampler
from tasks import get_task_sampler
import datetime

directory = "/tmpdir/m24047nmmr/in-context-learning/" #Put the path to the directory where the code is saved
run_dir = directory + "models/"

device = torch.device("cpu")
print("device : ", device)

n_layer = 12
n_ah = 8
n_deg= 1
#model = "2LMLP"
n_embed = 256
task = "polynomial_regression"


#Put the name of the model you want to test
#modele entrainé sur deg1 150k deg3 150k et 200K sur deg5 sans deg2 et deg4 sur U(-5,5)
#run_id = "1d_standard256_12l_8ah_deg_3_11-23_21-41" 
#run_id = "1d_standard256_12l_8ah_deg_1U1_02-06_22-50"

#N(0,1) deg1 Attention only sans mlp ni ADD NORM
#run_id = "1d_standard256_12l_8ah_deg_1N_02-11_18-29"
#run_id = "1d_standard256_1l_8ah_deg_1U1AL_02-11_15-13"
#run_id = "1d_standard256_12l_8ah_deg_1N_02-12_08-33"

#12L8AH FULL : N(0,10)
#run_id = "1d_standard256_12l_8ah_deg_1N10_02-12_18-46"
# FULL N(0,100)
#run_id = "1d_signclassification_12l_8ah_deg_1N1_03-03_09-23"


#AND
#run_id = "1d_signclassification_12l_8ah_deg_1N1_03-03_16-30"

#AND softmax: attention only
#run_id = "1d_softmaxAND_12l_8ah_deg_1NAttentionOnly_03-13_16-58"

#AND softmax: print attention weights 
run_id = "1d_softmaxANDwithAW_12l_8ah_deg_1NAttentionOnly_03-14_08-22"

#AND N(0,100)
#run_id = "1d_ANDsignclassification_12l_8ah_deg_1N100_03-05_10-03"

#AND N(0,1) full 40 points
#run_id = "1d_ANDsignclassification_12l_8ah_deg_1N141PTS_03-06_10-18"
#OR 
#run_id = "1d_ORsignclassification_12l_8ah_deg_1N1_03-04_14-18"

#50% OR 50% AND
#run_id = "1d_ANDsignclassification_12l_8ah_deg_1N15050_03-06_14-22"

#[TRUE,...,TRUE] if all positive [FALSE,...,FALSE] otherwise
#run_id = "1d_ANDsignclassification_12l_8ah_deg_1N1R1point_03-09_22-38"

#AND NEW ACTIV FT with a,b
#run_id = "1d_ANDnewactivation_12l_8ah_deg_1N_03-10_16-29"

#AND NEW ACTIV FT with lal, lbl
#run_id = "1d_ANDnewactivation_12l_8ah_deg_1N_03-10_20-54"

#AND NEW AVTI FT with exp(a), exp(b)
#run_id = "1d_ANDnewactivation_12l_8ah_deg_1N_03-11_01-19"

#OR NEW ACTIV FT with exp(a), exp(b)
#run_id = "1d_newactivationOR_12l_8ah_deg_1N_03-11_20-09"



print(run_id)
run_path = os.path.join(run_dir, task, run_id)
print("run_path : ", run_path)
task = "linear_classification"
model, conf = get_model_from_run(run_path,device=device) #Ici, on a model et conf
n_dims = conf.model.n_dims
print("n_dims", n_dims)
batch_size = conf.training.batch_size
print("batch size", batch_size)

data_sampler = get_data_sampler(conf.training.data, n_dims) 
task_sampler = get_task_sampler(
    "linear_classification",
    n_dims,
    batch_size,
    **conf.training.task_kwargs
)
task = task_sampler(max_dim=1)

#You can change the number of points to whatever you want
#xs = data_sampler.sample_xs(b_size=batch_size, n_points=conf.training.curriculum.points.end-1) #torch.Size([64, 40, 1])
xs = data_sampler.sample_xs(b_size=batch_size, n_points=40) 

#xs = data_sampler.sample_xs(b_size=batch_size, n_points=2)

#Put the distribution you want your x to be sampled from
list_l = []
list_scores =[]
for l in range(10,201,10):
    list_l.append(l)
    xs = data_sampler.sample_xs(b_size=batch_size, n_points=l) 
    scores_l = []
    for sigma in range(1,30):
        
        for i in range(xs.shape[0]):      
            for j in range(xs.shape[1]):    
                for k in range(xs.shape[2]):  
                    torch.manual_seed(100*i + 10*j +k)
                    xs[i, j, k] = sigma*torch.randn(1)  # Random value in N(0,1)

        positive_mask = xs > 0
        #AAAANNNDDD
        #Compute cumulative logical AND along each row
        result = positive_mask.cumprod(dim=1)
        z = result.float() * 2 - 1
        
        #OOORRRR
        # has_positive = torch.cumsum(positive_mask, dim=1) > 0
        # # Convert to 1 or -1
        # z = torch.where(has_positive, torch.tensor(1, dtype=torch.int8), torch.tensor(-1, dtype=torch.int8))
        
        
        #Task without autoregression
        # positive_mask = xs[:,:,0] > 0
        # result = torch.all(positive_mask, dim=1)  # Result will be [batch] (True/False)
        # # Now, expand the result to [batch, points] with either all True or all False
        # z = result.unsqueeze(1).expand(-1, xs.size(1)) 

        #If you want to do test only on x in U(-1,1) you change the range to 1
            
        with torch.no_grad():
            p = model(xs,z)    
            
        #b = 5
        ps =  p > 0 
        
        #for b in range(64):
        #print("xs", xs[b,:,0])
        #print("p", p[b,:])
        #print("z", z[b,:])
        #print("ps", ps[b,:])
        #print("z", z[b,:])
    
        
        wv = positive_mask.cumprod(dim=1).bool() #AND
        #wv = has_positive #OR
        #print("wv", wv[b,:,0] )

        #diff_indices = torch.nonzero(~torch.all(ps == wv[:,:,0], dim=1)).squeeze()
        #print("for sigma ", sigma, "Out of 64 batches, the number of incorrect ones are :", len(diff_indices.tolist()))
        
        diff_indices = torch.nonzero(~torch.all(ps[:,3:] == wv[:,3:,0], dim=1)).flatten()
        
        wv = wv.squeeze(-1)  # Shape becomes (64, 40)

        # Compare values from index 2 onwards for each batch
        comparison = ps[:, 3:] == wv[:, 3:]

        # Check if all values are equal for each batch
        result = comparison.all(dim=1)

        # Print batches where the values do not match
        mismatched_batches = torch.where(~result)[0]

        if mismatched_batches.numel() > 0:
            #print(f"Mismatch found in batches: {mismatched_batches.tolist()}")
            #print("for sigma", sigma,"the number of mismatch is:", len(mismatched_batches.tolist()))
            scores_l.append(len(mismatched_batches.tolist()))
        else:
            #print("For sigma", sigma, "All batches match from the third point onward.")
            scores_l.append(0)
            
    list_scores.append(scores_l)        
    
print("liste des l", list_l)
print("liste des scores", list_scores)
    
    #print("for sigma ", sigma, "the number of incorrect predictions are :", len(diff_indices), "out of 64 batches.")
    #print("pour le b",torch.nonzero(~torch.all(ps[b,3:] == wv[b,3:,0], dim=1)).flatten())


 #Generate the value for the heatmap why we vary both DtF and DtI in U(-sigma,sigma)
 #Choose the degree you want and then uncomment/comment corresponding to the degree   
##### Partie on prends 100 polynomes du type ax+b avec a,b dans N(0,1)




# print("p==wv", torch.equal(p,wv))
# for i in range(wv.shape[0]):
    
# diff_mask = (p != wv)  # Boolean mask of differences
# batch_indices = torch.nonzero(diff_mask.any(dim=1), as_tuple=True)[0]

# print("Batch indices where f and g differ:", batch_indices)


#     scores = []
#     sigmas = []
#     for sigma in range(1, 11):
#         mean_scores = 0
#         torch.manual_seed(sigma)
#         values = torch.randn(200)*sigma #N(0,sigma)
#         #values = -sigma + 2*sigma*torch.rand(200) #U(-sigma,sigma)    
#         coeff_a = values[:100]
#         coeff_b = values[100:]
#         #deg2
#         #coeff_c = values[200:300]
#         #deg3
#         #coeff_d = values[300:]
#         #deg4
#         #coeff_e = values[400:500]
#         #deg5
#         #coeff_f = values[500:]
#         #deg6
#         #coeff_g = values[600:]

#         for i in range(len(coeff_a)):
#             z = coeff_a[i] * xs + coeff_b[i] # torch.Size([64, 40, 1])
#             #z = coeff_c[i] * (xs**2) + coeff_a[i] * xs + coeff_b[i] 
#             #z =  coeff_d[i] * (xs**3) + coeff_c[i] * (xs**2) + coeff_a[i] * xs + coeff_b[i] 
#             #z = coeff_e[i] * (xs**4) + coeff_d[i] * (xs**3) + coeff_c[i] * (xs**2) + coeff_a[i] * xs + coeff_b[i] 
#             #z = coeff_f[i] * (xs**5) + coeff_e[i] * (xs**4) + coeff_d[i] * (xs**3) + coeff_c[i] * (xs**2) + coeff_a[i] * xs + coeff_b[i] 
#             #z = coeff_g[i] * (xs**6) + coeff_f[i] * (xs**5) + coeff_e[i] * (xs**4) + coeff_d[i] * (xs**3) + coeff_c[i] * (xs**2) + coeff_a[i] * xs + coeff_b[i] 
            
#             z = z[:,:,0] # torch.Size([64, 40])
#             #p = 0*xs[:,:,0]
#             with torch.no_grad():
#                 p = model(xs,z)
#             # p = torch.zeros_like(z)

#             # # Compute the 3-nearest neighbors for each batch
#             # for batch_idx in range(xs.size(0)):  # Iterate over each batch
#             #     batch_points = xs[batch_idx]  # Shape [40, 1]
                
#             #     # Compute pairwise distances within the batch
#             #     distances = torch.cdist(batch_points, batch_points)  # Shape [40, 40]
                
#             #     # Find the indices of the 3 nearest neighbors for each point
#             #     neighbors_idx = distances.argsort(dim=1)[:, 1:4]  # Skip the point itself (index 0)
                
#             #     # Compute the average of `ys` values for the 3 nearest neighbors
#             #     p[batch_idx] = z[batch_idx, neighbors_idx].mean(dim=1)
            
#             diff = z[:,2:] - p[:,2:]
#             mi = diff.square().mean()
#             mean_scores = mean_scores + (1/len(coeff_a))* mi
        
#         scores.append(mean_scores)
#         sigmas.append(sigma)
    
#     scoresx.append(scores) 
#     sigmasx.append(sigmas)       


# print("scores are:",scoresx)

