import json
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from pyclustering.cluster.kmedians import kmedians


IVF = defaultdict(list)

img_vecSet = []
# Load data from JSON file.
json_file_path = "D://CS//Data_System//resourses//img_vec.json"

with open(json_file_path, 'r') as json_file:
    print("Loading file...")
    entity_list = json.load(json_file)

    i = 1
    for entity in entity_list:
        img_id = entity['id']
        img_name = entity['img_name']
        vector = np.array(entity['array'])  
        img_vecSet.append(vector)

k = 40

initial_medians = random.sample(img_vecSet, k)

kmedians_instance = kmedians(img_vecSet, initial_medians)

kmedians_instance.process()

labels = kmedians_instance.get_clusters()

cluster_centers = kmedians_instance.get_medians()

# Print or use the cluster centers as needed
print("Cluster centers:")
print(cluster_centers)
print("Start collecting the results...")

for i in range(len(labels)):
    # Different
    IVF[i] = labels[i]


c_list = cluster_centers
IVF_res = []
for i in range(k):
    IVF_res.append({"id": i, "centroid": c_list[i], "data_pts": IVF[i]})


with open('D:\\CS\\Data_System\\Master Project\\local\\IVF\\Index_IVF\\index_file\\IVF_k-median.json', 'w') as json_file:
    json.dump(IVF_res, json_file, indent = 4)
