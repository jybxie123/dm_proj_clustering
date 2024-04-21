import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans


imgSet = []
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
        imgSet.append(vector)

sse = []
for k in range(1, 100):  # Possible K
    kmeans = MiniBatchKMeans(n_clusters=k, init='k-means++', batch_size=int(0.02*len(imgSet)))
    kmeans.fit(imgSet)
    sse.append(kmeans.inertia_)  # SSE for each K

plt.figure(figsize=(8, 6))
plt.plot(range(1, 100), sse, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('SSE')
plt.title('Elbow Method for Optimal K')
plt.show()
