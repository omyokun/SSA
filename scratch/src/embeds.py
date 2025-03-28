from collections import OrderedDict
import re
import os
import json
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
model = "standard"
n_embed = 256
task = "polynomial_regression"

#run_id = "1d_standard256_12l_8ah_deg_1N10_02-12_18-46" #Put the run_id of the model you want to use
#run_id ="1d_standard256_12l_8ah_deg_1N100_02-09_16-08"
#run_id = "1d_standard256_12l_8ah_deg_1N_02-11_18-29" #pas bon celui la

run_id = "1d_AHONLYWITHOUTMLPADDNORM256_1l_8ah_deg_1N1_02-26_18-45"




print(run_id)
run_path = os.path.join(run_dir, task, run_id)
print("run_path : ", run_path)
model, conf = get_model_from_run(run_path,device=device) #Ici, on a model et conf
n_dims = conf.model.n_dims
print("n_dims", n_dims)
batch_size = conf.training.batch_size
print("batch size", batch_size)
data_sampler = get_data_sampler(conf.training.data, n_dims)  # on lui donne str et return la classe

task_sampler = get_task_sampler(
    "toy_polynomial_regression",
    n_dims,
    batch_size,
    **conf.training.task_kwargs
)
task = task_sampler(max_dim=1)

#xs = data_sampler.sample_xs(b_size=batch_size, n_points=conf.training.curriculum.points.end-1) #torch.Size([64, 40, 1])


xs = data_sampler.sample_xs(b_size=batch_size, n_points=10) #torch.Size([64, 40, 1])

for i in range(xs.shape[0]):        # Parcourir la première dimension
    for j in range(xs.shape[1]):    # Parcourir la deuxième dimension
        for k in range(xs.shape[2]):  # Parcourir la troisième dimension
            torch.manual_seed(50*i + 10*j +k)
            torch.manual_seed(100*i + 10*j +k)

            xs[i, j, k] = 1000*j

print("xs",xs[8,:,:])            
#ys = xs[:,:,0]
ys = task.evaluate(xs) #torch.Size([64, 40])
with torch.no_grad():
    pred, inn, output = model(xs,ys) 
    #pred = model(xs,ys) 
 
print("pred",pred[8,:])

 
output_path = "embeddings.json"
embeds_to_save = inn[8, :, :].tolist()

output_pathdecoder = "embeddingsdecoder.json"
decoderembeds_to_save = output[8, :, :].tolist()






from sklearn.metrics.pairwise import cosine_similarity

# Convert to numpy array if needed
import numpy as np
embeds_to_save2 = np.array(embeds_to_save)

# Compute cosine similarity
similarity_matrix = cosine_similarity(embeds_to_save)
print(similarity_matrix)
np.savetxt("similarity_matrix.txt", similarity_matrix, fmt="%.6f")


sns.heatmap(similarity_matrix, annot=False, cmap="viridis")
plt.xticks(ticks=np.arange(0,9), labels=np.arange(0,9))
plt.yticks(ticks=np.arange(0,9), labels=np.arange(0,9))
plt.xlabel(" [1..10]")
plt.ylabel(" [1..10]")
plt.savefig("/tmpdir/m24047nmmr/in-context-learning/src/htmapembedings10.png")
plt.show()


with open(output_path, "w") as f:
    json.dump(embeds_to_save, f, indent=4)

print(f"Embeddings saved to {output_path}")

with open(output_pathdecoder, "w") as f:
    json.dump(decoderembeds_to_save, f, indent=4)

print(f"Embeddings saved to {output_pathdecoder}")


# from sklearn.manifold import TSNE
# def plot_tsne(embeddings, labels, title, figure_num):
#     tsne = TSNE(n_components=2, random_state=42)
#     reduced_embeddings = tsne.fit_transform(embeddings)

#     plt.figure(figure_num, figsize=(10, 8))
#     plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=20, c="red")
#     for i in range(xs.shape[1]):
#         plt.annotate(str(xs[8,i,0].item()), (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=8, alpha=0.95)

#     for i, label in enumerate(labels):
#         plt.text(
#             reduced_embeddings[i, 0],
#             reduced_embeddings[i, 1],
#             label,
#             fontsize=8
#         )

#     plt.title(title)
#     plt.show()

# plot_tsne(embeds_to_save," ", " ",1)
 
