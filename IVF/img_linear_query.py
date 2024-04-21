import os
import shutil
import json
import heapq

resource_path = "D:\\CS\Data_System\\resourses\JPEGImages"


# img_name = "2007_000129.jpg" # cyclist + cycle
img_name = "2007_000068.jpg"
file_path = os.path.join(resource_path, img_name)



json_file_path = "D://CS//Data_System//resourses//img_vec.json"  
json_file = open(json_file_path, 'r')
entity_list = json.load(json_file)
img_vector = []
for entity in entity_list:
    if entity["img_name"] == img_name:
        img_vector = entity["array"]

print(img_vector)



print("Searching Image...")
def dist(v1, v2):
    distance = 0
    for i in range(len(v1)):
        distance += (v1[i] - v2[i])**2
    return distance


def linear_query():
    min_heap = []
    topk = 10
    for entity in entity_list:
        if entity["img_name"] != img_name:
            distance = dist(img_vector, entity["array"])
            item = (-distance, entity)
            
            heapq.heappush(min_heap, item)
            if len(min_heap) > topk:
                heapq.heappop(min_heap)

    return min_heap



if img_vector == []:
    # Empty
    raise KeyboardInterrupt("Error: Empty image vector!!!")

linear_query_res = linear_query()

result_path = "D:\\CS\\Data_System\\Master Project\\local\\IVF\\results"

def clear_dir_addRes(res_dir_path, res_list):
    # Clear the result directory and add results.
    file_list = os.listdir(res_dir_path)
    if len(file_list):
        # Clear the directory.
        for filename in file_list:
            file_path = os.path.join(res_dir_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    for entity in res_list:
        # print(entity)
        p = os.path.join(resource_path, entity[1]["img_name"])
        print(p)
        shutil.copy(p, res_dir_path)

clear_dir_addRes(result_path, linear_query_res)
# clear_dir_addRes(result_path_topD, query_res_topD)

def recall(res, GT_res):
    # Calculate the recall with Ground Truth.
    cnt = 0
    GT_res_img = []
    for r in GT_res:
        if r[1]["img_name"] != img_name:
            GT_res_img.append(r[1]["img_name"])
    for r in res:
        if r[1]["img_name"] in GT_res_img:
            cnt += 1
    return cnt / len(GT_res_img)

# recall_topD = recall(query_res_topD, linear_query_res)
# print(recall_topD)
# print(linear_query_res)
# print(linear_query_res[0])


json_file.close()
