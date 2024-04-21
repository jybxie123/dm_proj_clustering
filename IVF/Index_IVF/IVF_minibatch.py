import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import MiniBatchKMeans

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
kmeans = MiniBatchKMeans(n_clusters=k, init='k-means++', batch_size=int(0.02*len(img_vecSet)))
labels = kmeans.fit_predict(img_vecSet)

# Get the coordinates of each cluster center
cluster_centers = kmeans.cluster_centers_

# Print or use the cluster centers as needed
print("Cluster centers:")
print(cluster_centers)
print("Start collecting the results...")
for i in range(len(labels)):
    label = str(labels[i])
    IVF[label].append(i)

c_list = kmeans.cluster_centers_.tolist()
IVF_res = []
for i in range(k):
    IVF_res.append({"id": i, "centroid": c_list[i], "data_pts": IVF[str(i)]})


with open('D:\\CS\\Data_System\\Master Project\\local\\IVF\\Index_IVF\\index_file\\IVF_minibatch.json', 'w') as json_file:
    json.dump(IVF_res, json_file, indent = 4)


