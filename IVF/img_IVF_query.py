import os
import shutil
import json
import heapq
import random
resource_path = "D:\\CS\Data_System\\resourses\JPEGImages"


# img_name = "2007_000129.jpg" # cyclist + cycle
# img_name = "2008_001429.jpg" # poor performance
# img_name = "2008_001140.jpg" 
# img_name = "2008_000620.jpg" # poor performance

img_list = ["2008_000401.jpg", "2007_005896.jpg", "2007_002954.jpg", "2007_000175.jpg", "2007_006317.jpg", "2007_003910.jpg", "2007_002789.jpg", "2007_004538.jpg", "2007_002216.jpg", "2007_007432.jpg"]
random.seed(42)

def query_recall(img_name):
    print(img_name)
    json_file_path = "D://CS//Data_System//resourses//img_vec.json"  
    json_file = open(json_file_path, 'r')
    entity_list = json.load(json_file)
    # file_path = os.path.join(resource_path, img_name)
    resource_path = "D:\\CS\Data_System\\resourses\JPEGImages"
    img_vector = []
    for entity in entity_list:
        if entity["img_name"] == img_name:
            img_vector = entity["array"]

    # print(img_vector)



    print("Searching Image...")
    def dist(v1, v2):
        distance = 0
        for i in range(len(v1)):
            distance += (v1[i] - v2[i])**2
        return distance


    def linear_query(candidate_list):
        min_heap = []
        topk = 100
        for entity in candidate_list:
            if entity["img_name"] != img_name:
                distance = dist(img_vector, entity["array"])
                item = (-distance, entity)
                try:
                    heapq.heappush(min_heap, item) # Sometimes the distances are the same, than it will compare the second value which is a dict (uncomparable type).
                except:
                    print("Exception.") 
                    
                    item_sub = (item[0] - 0.00000000001*random.random(), item[1])
                    heapq.heappush(min_heap, item_sub)
                if len(min_heap) > topk:
                    try:
                        heapq.heappop(min_heap) 
                    except:
                        print("Exception.")
        result = sorted(min_heap, key=lambda x: -x[0])
        return result[:topk]

    def IVF_query(idx_path):
        nprobe = 10
        k = 40
        cluster_min_heap = []
        with open(idx_path, 'r') as ivf_fd:
            ivf_data = json.load(ivf_fd)
            for entity in ivf_data:
                distance = dist(img_vector, entity["centroid"])
                item = (-distance, entity)
                heapq.heappush(cluster_min_heap, item)
                if len(cluster_min_heap) > nprobe:
                    heapq.heappop(cluster_min_heap)
        
        candidate_list = []
        for c in cluster_min_heap:
            candidate_list.extend(c[1]["data_pts"]) # Index of candidate image.

        for i in range(len(candidate_list)):
            candidate_list[i] = entity_list[candidate_list[i]]

        


        return linear_query(candidate_list)




    if img_vector == []:
        # Empty
        raise KeyboardInterrupt("Error: Empty image vector!!!")

    linear_query_res = linear_query(entity_list)
    IVF_k_means_res = IVF_query("D:\\CS\\Data_System\\Master Project\\local\\IVF\\Index_IVF\\index_file\\IVF_k-means.json")
    IVF_k_meansPlusPlus_res = IVF_query("D:\\CS\\Data_System\\Master Project\\local\\IVF\\Index_IVF\\index_file\\IVF_k-means++.json")
    IVF_k_median_res = IVF_query("D:\\CS\\Data_System\\Master Project\\local\\IVF\\Index_IVF\\index_file\\IVF_k-median.json")
    IVF_minibatch_res = IVF_query("D:\\CS\\Data_System\\Master Project\\local\\IVF\\Index_IVF\\index_file\\IVF_minibatch.json")

    result_path = "D:\\CS\\Data_System\\Master Project\\local\\IVF\\results"
    result_path_IVF_k_means = "D:\\CS\\Data_System\\Master Project\\local\\IVF\\results_IVF_k-means"
    result_path_IVF_k_meansPlusPlus = "D:\\CS\\Data_System\\Master Project\\local\\IVF\\results_IVF_k-meansPlusPlus"
    result_path_IVF_k_median = "D:\\CS\\Data_System\\Master Project\\local\\IVF\\results_IVF_k-median"
    result_path_IVF_minibatch = "D:\\CS\\Data_System\\Master Project\\local\\IVF\\results_IVF_minibatch"


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
            # print(p)
            shutil.copy(p, res_dir_path)

    clear_dir_addRes(result_path, linear_query_res)
    clear_dir_addRes(result_path_IVF_k_means, IVF_k_means_res)
    clear_dir_addRes(result_path_IVF_k_meansPlusPlus, IVF_k_meansPlusPlus_res)
    clear_dir_addRes(result_path_IVF_k_median, IVF_k_median_res)
    clear_dir_addRes(result_path_IVF_minibatch, IVF_minibatch_res)


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

    recall_1 = recall(IVF_k_means_res, linear_query_res)
    recall_2 = recall(IVF_k_meansPlusPlus_res, linear_query_res)
    recall_3 = recall(IVF_k_median_res, linear_query_res)
    recall_4 = recall(IVF_minibatch_res, linear_query_res)
    print("k-means", recall_1)
    print("k-means++", recall_2)
    print("k-median", recall_3)
    print("minibatch", recall_4)

    json_file.close()
    return recall_1, recall_2, recall_3, recall_4

recall_a = 0
recall_b = 0
recall_c = 0
recall_d = 0
for img in img_list:
    delta_a, delta_b, delta_c, delta_d = query_recall(img)
    recall_a += delta_a / len(img_list)
    recall_b += delta_b / len(img_list)
    recall_c += delta_c / len(img_list)
    recall_d += delta_d / len(img_list)

print("----- Test Results -----")
print("k-means", recall_a)
print("k-means++", recall_b)
print("k-median", recall_c)
print("minibatch", recall_d)
# query_recall("2007_005896.jpg")