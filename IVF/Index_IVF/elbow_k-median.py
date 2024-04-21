'''
    It takes a long time to execute...
'''









import json
import numpy as np
import random
import matplotlib.pyplot as plt
from pyclustering.cluster.kmedians import kmedians
from pyclustering.utils.metric import distance_metric, type_metric

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


for k in range(1,3):
    np.random.seed(42)
    initial_medians = random.sample(imgSet, k)
    print(initial_medians)

    kmedians_instance = kmedians(imgSet, initial_medians)

    kmedians_instance.process()

    sse.append(kmedians_instance.get_total_wce())
    # clusters = kmedians_instance.get_clusters()
    # medians = kmedians_instance.get_medians()
    # medians_array = np.array(medians)   

plt.figure(figsize=(8, 6))
plt.plot(range(1, 3), sse, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('SSE')
plt.title('Elbow Method for Optimal K')
plt.show()
