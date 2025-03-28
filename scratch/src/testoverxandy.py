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
#run_id ="1d_standard256_12l_8ah_deg_1N100_02-09_16-08"

#18L8AH deg1
#run_id = "1d_standard256_18l_8ah_deg_1N1_02-14_06-08"

#N(0,1) nv Actif ft
run_id = "1d_ANDnewactivationLF_12l_8ah_deg_1N_03-11_09-21"
#run_id = "1d_SSALFwithAW_12l_8ah_deg_1NFullTransformer_03-20_23-26"



#run_id = "1d_standard64_12l_8ah_deg_1_02-04_11-30"


#run_id = "1d_softmaxLFwithAW_12l_8ah_deg_1NAttentionOnly_03-16_16-38"
#run_id = "1d_softmaxLFwithAW_12l_8ah_deg_1NFullTransformer_03-18_12-37" #DEG 1 Linear function full transformer

print(run_id)
run_path = os.path.join(run_dir, task, run_id)
print("run_path : ", run_path)
model, conf = get_model_from_run(run_path,device=device) #Ici, on a model et conf
n_dims = conf.model.n_dims
print("n_dims", n_dims)
batch_size = conf.training.batch_size
print("batch size", batch_size)

data_sampler = get_data_sampler(conf.training.data, n_dims) 
task_sampler = get_task_sampler(
    "polynomial_regression",
    n_dims,
    batch_size,
    **conf.training.task_kwargs
)
task = task_sampler(max_dim=1)

#You can change the number of points to whatever you want
#xs = data_sampler.sample_xs(b_size=batch_size, n_points=conf.training.curriculum.points.end-1) #torch.Size([64, 40, 1])
xs = data_sampler.sample_xs(b_size=batch_size, n_points=9)

#Put the distribution you want your x to be sampled from
for i in range(xs.shape[0]):      
    for j in range(xs.shape[1]):    
        for k in range(xs.shape[2]):  
            torch.manual_seed(100*i + 10*j +k)
            #xs[i, j, k] = -1 + 2*torch.rand(1)  # Random value in U(-1,1)
            xs[i, j, k] = torch.randn(1)  # Random value in N(0,1)


#xs[:,0,0]= 1 
#xs[:,3,0] = 100
#xs[:,1,0] = 101
#xs[:,2,0] = 102
# xs[:,3,0] = 103         
# xs[:,4,0] = 104
# xs[:,5,0] = 105


########UNCOMMENT IF YOU WANT TO SEE ATTENTION WEIGHTS
# z = xs[:,:,0]
# with torch.no_grad():
#     #p, predic = model(xs,z)  
#     predic = model(xs,z)  
    
#print("ppppp", predic)
# #print("p", p.shape)
# attentions = p.attentions  # Shape: (num_layers, batch_size, num_heads, seq_len, seq_len)

# # Print attention weights of the last layer
# last_layer_attention = attentions[-1]  # Last layer attention weights
# print("attenions", str(last_layer_attention.shape))
# with open("/tmpdir/m24047nmmr/in-context-learning/src/attention.txt","a") as file:
#     file.write(str(last_layer_attention[5,:,:].tolist())+'\n\n' + '\n')

# #print("Last layer attention weights shape:", last_layer_attention.shape)
# print(last_layer_attention)
# print("xs", xs[5,:,0])
# print("pred", predic[5,:])
##################



# ###### TO UNCOMMENT AFTER FOR TESTS OVER SIGMAS/XS


 #Generate the value for the heatmap why we vary both DtF and DtI in U(-sigma,sigma)
 #Choose the degree you want and then uncomment/comment corresponding to the degree   
##### Partie on prends 100 polynomes du type ax+b avec a,b dans N(0,1)
scoresx = []
sigmasx = []

#If you want to do test only on x in U(-1,1) you change the range to 1
for sigmax in range(1,11):
    for i in range(xs.shape[0]):      
        for j in range(xs.shape[1]):    
            for k in range(xs.shape[2]):  
                torch.manual_seed(100*i + 10*j +k)
                #xs[i, j, k] = -sigmax + 2*sigmax*torch.rand(1) 
                xs[i, j, k] = sigmax*torch.randn(1) 
    
    #xs[:,5,0]= 20000
    
    scores = []
    sigmas = []
    for sigma in range(1,11):
        mean_scores = 0
        torch.manual_seed(sigma)
        values = torch.randn(200)*sigma #N(0,sigma)
        #values = -sigma + 2*sigma*torch.rand(200) #U(-sigma,sigma)    
        coeff_a = values[:100]
        coeff_b = values[100:]
        #deg2
        #coeff_c = values[200:300]
        #deg3
        #coeff_d = values[300:]
        #deg4
        #coeff_e = values[400:500]
        #deg5
        #coeff_f = values[500:]
        #deg6
        #coeff_g = values[600:]

        for i in range(len(coeff_a)):
            #z = xs
            z = coeff_a[i] * xs + coeff_b[i] # torch.Size([64, 40, 1])
            #z = coeff_c[i] * (xs**2) + coeff_a[i] * xs + coeff_b[i] 
            #z =  coeff_d[i] * (xs**3) + coeff_c[i] * (xs**2) + coeff_a[i] * xs + coeff_b[i] 
            #z = coeff_e[i] * (xs**4) + coeff_d[i] * (xs**3) + coeff_c[i] * (xs**2) + coeff_a[i] * xs + coeff_b[i] 
            #z = coeff_f[i] * (xs**5) + coeff_e[i] * (xs**4) + coeff_d[i] * (xs**3) + coeff_c[i] * (xs**2) + coeff_a[i] * xs + coeff_b[i] 
            #z = coeff_g[i] * (xs**6) + coeff_f[i] * (xs**5) + coeff_e[i] * (xs**4) + coeff_d[i] * (xs**3) + coeff_c[i] * (xs**2) + coeff_a[i] * xs + coeff_b[i] 
            
            z = z[:,:,0] # torch.Size([64, 40])
            #p = 0*xs[:,:,0]
            with torch.no_grad():
                p = model(xs,z)
            # p = torch.zeros_like(z)
            
            # # Compute the 3-nearest neighbors for each batch
            # for batch_idx in range(xs.size(0)):  # Iterate over each batch
            #     batch_points = xs[batch_idx]  # Shape [40, 1]
                
            #     # Compute pairwise distances within the batch
            #     distances = torch.cdist(batch_points, batch_points)  # Shape [40, 40]
                
            #     # Find the indices of the 3 nearest neighbors for each point
            #     neighbors_idx = distances.argsort(dim=1)[:, 1:4]  # Skip the point itself (index 0)
                
            #     # Compute the average of `ys` values for the 3 nearest neighbors
            #     p[batch_idx] = z[batch_idx, neighbors_idx].mean(dim=1)
            
            diff = z[:,3:] - p[:,3:]
            mi = diff.square().mean()
            mean_scores = mean_scores + (1/len(coeff_a))* mi
        
        scores.append(mean_scores)
        sigmas.append(sigma)
        
        #print("xs", xs[3,:,0])
        #print("predictions", p[3,:])
    
    scoresx.append(scores) 
    sigmasx.append(sigmas)       


print("scores are:",scoresx)