import time
from pyspark import SparkContext
import os
import sys
import csv
import math
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
from copy import deepcopy

os.environ['PYSPARK_PYTHON'] = '/Users/xinmengqiao/opt/anaconda3/envs/py36/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/Users/xinmengqiao/opt/anaconda3/envs/py36/bin/python'


def hash_index(index):
    return ((133333 * index + 11) % 23333333) % 5


def assign_cluster(input_index, input_cluster):
    assigned_info = defaultdict(list)
    for i in range(len(input_index)):
        cluster_id = input_cluster[i]
        assigned_info[cluster_id].append(input_index[i])
    return assigned_info


def kmeans_assign(n, index_lst, sample):
    values = []
    for h in index_lst:
        values.append(sample[h])
    cls = KMeans(n_clusters=n).fit(values)
    res = cls.labels_
    return res


def initialize_sets(current_cluster, sample):
    get_assigned = set()
    result_dic = dict()
    for c_idx in current_cluster.keys():
        points_idx_lst = current_cluster[c_idx]
        points_vals = []
        for point in points_idx_lst:
            points_vals.append(sample[point])
        points_vals = np.array(points_vals)

        n = len(points_vals)
        sums = points_vals.sum(axis=0)
        sum_sq = (points_vals ** 2).sum(axis=0)
        # print(n, len(sums), len(sum_sq))

        if n <= 1:
            for need_idx in points_idx_lst:
                retain_glb[need_idx] = sample[need_idx]
        else:
            result_dic[c_idx] = [points_idx_lst, n, sums, sum_sq]

            get_assigned = get_assigned.union(set(points_idx_lst))

    return result_dic, get_assigned


def generate_cs_from_rs(retain_index, cluster_number, sample_i):
    if len(retain_index) < cluster_number:
        a = len(retain_index)
        return
    else:
        res_3 = kmeans_assign(cluster_number, retain_index, retain_glb)
        # print('333', res_3)
        cs_cluster = assign_cluster(retain_index, res_3)
        additional_cs, remove_from_rs = initialize_sets(cs_cluster, retain_glb)

        # combine the original compression set with the newly generated compression set from retain set
        new_key = len(compress_glb)
        for k in additional_cs.keys():
            compress_glb[new_key] = additional_cs[k]
            new_key += 1
    return remove_from_rs


def mahalanobis(cluster, new_point):
    new_point = np.array(new_point)
    n = cluster[1]
    sums = np.array(cluster[2])
    sum_sq = np.array(cluster[3])

    centroid = sums/n
    std = np.sqrt((sum_sq/n) - centroid**2)
    dist = np.sqrt((((new_point - centroid) / std) ** 2).sum())
    return dist


def assign_pt_cluster(curr_pt, sample):
    min_dist = math.inf
    cluster_assign = None
    values = np.array(sample[curr_pt])
    d = len(values)
    theta = 2 * np.sqrt(d)
    # print(theta)
    # loop through discard set first
    for k in discard_glb.keys():
        distance = mahalanobis(discard_glb[k], values)
        if distance < min_dist:
            min_dist = distance
            if distance <= theta:
                cluster_assign = k
    # print(min_dist)
    # print(discard_glb[cluster_assign][1:])

    # assign to cluster if distance under threshold
    if cluster_assign is not None:
        discard_glb[cluster_assign][0].append(curr_pt)
        discard_glb[cluster_assign][1] += 1
        discard_glb[cluster_assign][2] += values
        discard_glb[cluster_assign][3] += values ** 2
        # print(discard_glb[cluster_assign][1:])
        return cluster_assign, 'ds'

    if not cluster_assign:
        # print('hahaha')
        for k in compress_glb.keys():
            distance = mahalanobis(compress_glb[k], values)
            if distance < min_dist:
                min_dist = distance
                if distance <= theta:
                    cluster_assign = k
                    # print(distance, theta)

        if cluster_assign is not None:
            compress_glb[cluster_assign][0].append(curr_pt)
            compress_glb[cluster_assign][1] += 1
            compress_glb[cluster_assign][2] += values
            compress_glb[cluster_assign][3] += values ** 2
            # print(compress_glb[cluster_assign][1:])
        else:
            retain_glb[curr_pt] = sample[curr_pt]
            # print(len(retain_glb))
    return cluster_assign, 'non-ds'


def cs_merge(compress):
    cs_val = list(compress.values())
    combined = set()

    for i in range(len(cs_val)):
        if i in combined:
            continue

        current_cls = cs_val[i]

        ids1 = current_cls[0]
        num1 = current_cls[1]
        n_sum1 = current_cls[2]
        sum_sq1 = current_cls[3]
        centroid = n_sum1 / num1
        dimension = len(centroid)
        threshold = 2 * np.sqrt(dimension)

        for j in range(len(cs_val)):
            if j == i or j in combined:
                continue
            cluster_2 = cs_val[j]
            dist = mahalanobis(cluster_2, centroid)

            if dist <= threshold:
                ids2 = cluster_2[0]
                ids1.extend(ids2)
                num1 += cluster_2[1]
                n_sum1 += np.array(cluster_2[2])
                sum_sq1 += np.array(cluster_2[3])
                centroid = n_sum1 / num1

                combined.add(j)
        # after the inner for loop finish, update the values in the list
        cs_val[i][0] = ids1
        cs_val[i][1] = num1
        cs_val[i][2] = n_sum1
        cs_val[i][3] = sum_sq1

    # discard those clusters in combined set
    new_lst = [cs_val[val] for val in range(len(cs_val)) if val not in combined]
    new_cs = {}
    for k, v in enumerate(new_lst):
        new_cs[k] = v
    return new_cs


def intermediate(round):
    num_dis = 0
    for ks, vs in discard_glb.items():
        num_dis += len(vs[0])

    cm_num = len(compress_glb)

    num_cm = 0
    for ks, vs in compress_glb.items():
        num_cm += len(vs[0])

    num_rt = len(retain_glb)

    return [num_dis, cm_num, num_cm, num_rt]


if __name__ == '__main__':
    input_file_path = sys.argv[1]
    n_cluster = int(sys.argv[2])
    output_file_path = sys.argv[3]

    start = time.time()
    sc = SparkContext().getOrCreate()
    sc.setLogLevel("ERROR")
    rdd = sc.textFile(input_file_path)

    all_points = rdd.map(lambda x: (int(x.split(',')[0]), [float(a) for a in x.split(',')[2:]])).persist()
    sample_0 = all_points.filter(lambda x: hash_index(x[0]) == 0).collectAsMap()

    # alls = all_points.collect()
    # print(len(alls))
    # print(len(sample_0))

    discard_glb = dict()
    retain_glb = dict()
    compress_glb = dict()
    inter_res = []

    indexs = list(sample_0.keys())
    res = kmeans_assign(n_cluster * 7, indexs, sample_0)

    # Step 2. Run K-Means with a large K (e.g., 5 times of the number of the input clusters)
    cluster_1 = defaultdict(list)
    keeps = set()
    for i in range(len(indexs)):
        clus_idx = res[i]
        cluster_1[clus_idx].append(indexs[i])
        if len(cluster_1[clus_idx]) > 1:
            keeps = keeps.union(set(cluster_1[clus_idx]))

    # Step 3. In the K-Means result from Step 2, move all the clusters that contain only one point to RS
    for keys in set(indexs):
        if keys not in keeps:
            retain_glb[keys] = sample_0[keys]
    print(len(retain_glb), len(keeps))

    # Step 4. Run K-Means again to cluster the rest of the data points with K = the number of input clusters
    keep_idx = list(keeps)
    res_2 = kmeans_assign(n_cluster, keep_idx, sample_0)
    initial_cluster = assign_cluster(keep_idx, res_2)
    # print(res_2[:10])

    # Step 5. Use the K-Means result from Step 4 to generate the DS clusters
    # Now DS is initialized
    discard_glb, _ = initialize_sets(initial_cluster, sample_0)

    # print(discard_glb.keys())
    # print(discard_glb[0][1:])
    # print(retain_glb)

    # Step 6. Run K-Means on the points in the RS with a large K, to generate CS (clusters with more than one points)
    # and RS (clusters with only one point).
    retain_index_ls = list(retain_glb.keys())
    remove_rs = generate_cs_from_rs(retain_index_ls, n_cluster * 7, sample_0)
    if remove_rs:
        for need_remove in remove_rs:
            retain_glb.pop(need_remove)

    # print('cs', compress_glb)
    # print(retain_glb)
    inter_res.append(intermediate(1))

    for ld in range(1, 5):
        # Step 7. Load another 20% of the data randomly.
        sample_2 = all_points.filter(lambda x: hash_index(x[0]) == ld).collectAsMap()

        # Step 8. For the new points, compare them to each of the DS using the Mahalanobis Distance and assign
        # them to the nearest DS clusters if the distance is < 2std

        ct, ct2 = 0, 0
        for pt in sample_2.keys():
            a, b = assign_pt_cluster(pt, sample_2)
            if b == 'ds':
                ct += 1
            else:
                ct2 += 1

        # print(ct, ct2)
        # print('after assigned', len(retain_glb))

        # Step 11. Run K-Means on the RS with a large K
        retain_idx = list(retain_glb.keys())

        remove_rs = generate_cs_from_rs(retain_idx, n_cluster * 5, sample_2)
        # print(remove_rs)
        if remove_rs:
            for need_remove in remove_rs:
                retain_glb.pop(need_remove)
        # print(len(retain_glb))

        # Step 12. Merge CS clusters that have a Mahalanobis Distance < 2std.
        compress_glb = cs_merge(compress_glb)

        inter_res.append(intermediate(ld+1))
    # for final step, merge cs to ds
    remove_keys = set()
    for ks, vs in compress_glb.items():
        min_dist = math.inf
        merged_ds = None

        ids1 = vs[0]
        num1 = vs[1]
        n_sum1 = vs[2]
        sum_sq1 = vs[3]
        centroid = n_sum1 / num1
        dimension = len(centroid)

        for ds, vds in discard_glb.items():
            distn = mahalanobis(vds, centroid)
            if distn < min_dist:
                min_dist = distn
                merged_ds = ds

        if min_dist < 2 * np.sqrt(dimension):
            discard_glb[ds][0].extend(ids1)
            discard_glb[ds][1] += num1
            discard_glb[ds][2] += n_sum1
            discard_glb[ds][3] += sum_sq1

            remove_keys.add(ks)
    for rm in remove_keys:
        compress_glb.pop(rm)

    final_res = []
    for label in discard_glb.keys():
        for point in discard_glb[label][0]:
            final_res.append((point, label))

    for label in compress_glb.keys():
        for point in compress_glb[label][0]:
            final_res.append((point, -1))

    for point in retain_glb.keys():
        final_res.append((point, -1))

    final_res = sorted(final_res, key=lambda x: x[0])

    with open(output_file_path, 'w') as out:
        out.write('The intermediate results:\n')
        for i in range(len(inter_res)):
            x = inter_res[i]
            line = f'Round {i + 1}: {x[0]},{x[1]},{x[2]},{x[3]}\n'
            out.write(line)
        out.write('\n')
        out.write('The clustering results:\n')
        for x in final_res:
            out.write(f'{x[0]},{x[1]}\n')
    out.close()

    end = time.time()
    print(end - start)
