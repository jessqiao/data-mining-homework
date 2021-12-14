import time
from pyspark import SparkContext
import os
from collections import defaultdict
import sys
import csv

# os.environ['PYSPARK_PYTHON'] = '/Users/xinmengqiao/opt/anaconda3/envs/py36/bin/python'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/Users/xinmengqiao/opt/anaconda3/envs/py36/bin/python'


def edge_cal(vertax, vertax_info, theta):
    edge_info = []
    uid = vertax[0]
    u_set = set(vertax[1])

    for v in vertax_info:
        vid = v[0]
        v_set = set(v[1])

        if uid != vid:
            if len(u_set & v_set) >= theta:
                edge_info.append(vid)
    return tuple([uid, edge_info])


def calculate_betweeness(current_node):
    # initialize dictionary to keep node:[parent_node 1, parent_node 2...]
    parent_dic = defaultdict(list)
    # initialize dictionary to keep level:[node 1, node 2...]
    level_dic = defaultdict(list)
    # initialize dictionary to keep node:[credit]
    credit_dic = dict()
    used_set = set()

    root = current_node

    bfs_q = [root]
    level = 0
    level_dic[level] = root
    credit_dic[root] = 1
    used_set.add(root)
    parent_dic[root] = []
    next_level_q = []

    # use queue to track BFS for every level. Put children in the next_level queue first.
    while bfs_q or next_level_q:
        level += 1
        while bfs_q:
            de_q = bfs_q.pop(0)

            for children in adj_dict[de_q]:
                if children not in used_set:
                    parent_dic[children].append(de_q)
                    next_level_q.append(children)

            next_level_q = list(set(next_level_q))
        # After the current level queue is empty, dequeue the next_level_q and enqueue all to the BFS queue
        while next_level_q:
            a = next_level_q.pop(0)
            bfs_q.append(a)
            used_set.add(a)
            a_credit = 0
            for parent in parent_dic[a]:
                a_credit += credit_dic[parent]
            credit_dic[a] = a_credit
            level_dic[level].append(a)

    bet_dic = {}
    edge_bet = {}

    # calculate betweeness for every edge on the BFS graph
    while level > 0:
        level -= 1
        for node in level_dic[level]:
            bet_dic[node] = bet_dic.get(node, 0) + 1
            for parent_node in parent_dic[node]:
                weight = credit_dic[parent_node] / credit_dic[node]
                edge_bet[tuple(sorted([node, parent_node]))] = bet_dic[node] * weight
                bet_dic[parent_node] = bet_dic.get(parent_node, 0) + bet_dic[node] * weight

    return [(edge, val) for edge, val in edge_bet.items()]


def find_connected_nodes(start_element):
    # use BFS queue to traverse all connected nodes until no more
    bfs_queue = [start_element]
    traverse_used = set()
    traverse_used.add(start_element)
    while bfs_queue:
        a = bfs_queue.pop(0)
        for children in adj_dict[a]:
            if children not in traverse_used:
                bfs_queue.append(children)
                traverse_used.add(children)
    return traverse_used


def find_current_communities(v_lst):
    # find new communities
    communities = []
    all_nodes = set(v_lst)
    used_node = set()
    remain_nodes = all_nodes - used_node

    while remain_nodes:
        start_elem = remain_nodes.pop()
        used_node.add(start_elem)
        # start with a random point, and then return all the nodes connected to the random node as one community
        new_community = find_connected_nodes(start_elem)
        communities.append(new_community)
        used_node = used_node.union(new_community)
        remain_nodes = all_nodes - used_node

    return communities


def calculate_modularity(all_communities):
    overall_mod = 0
    for comm in all_communities:
        mod = 0
        for v1 in comm:
            for v2 in comm:
                a_12 = Adj.get(tuple(sorted([v1, v2])), 0)
                mod += a_12 - (degree[v1] * degree[v2]) / (2 * m)
        overall_mod += mod
    overall_mod = overall_mod / (2 * m)
    return overall_mod


if __name__ == '__main__':
    theta = int(sys.argv[1])
    input_file_path = sys.argv[2]
    output_1 = sys.argv[3]
    output_2 = sys.argv[4]

    sc = SparkContext().getOrCreate()
    sc.setLogLevel("ERROR")

    start = time.time()
    raw_rdd = sc.textFile(input_file_path)
    file_title = raw_rdd.first()
    rdd = raw_rdd.filter(lambda row: row != file_title).map(
        lambda x: (x.split(',')[0], x.split(',')[1])).distinct()

    graph_rdd = rdd.map(lambda x: (x[0], [x[1]])).reduceByKey(lambda x, y: x+y)\
        .map(lambda x: (x[0], list(set(x[1])))).persist()

    v_info = graph_rdd.collect()


    g_rdd = graph_rdd.map(lambda x: edge_cal(x, v_info, theta)).filter(lambda x: len(x[1]) > 0).persist()

    # get the edges, (a, b), (b, a) only keep (a, b)
    edge_lst = g_rdd.flatMap(lambda x: [(x[0], val) for val in x[1]])\
        .filter(lambda x: x[0] < x[1]).distinct().collect()

    # get the vertices
    v_lst = g_rdd.map(lambda x: x[0]).distinct().collect()

    # get connected nodes {uid : [vid1, vid2, vid3....]}
    adj_dict = g_rdd.collectAsMap()

    # calculate initial betweeness for each node as the root
    betweeness = g_rdd.map(lambda x: x[0]).distinct().map(lambda x: calculate_betweeness(x))\
        .flatMap(lambda x: [edge for edge in x]).reduceByKey(lambda x, y: x+y)\
        .map(lambda x: (x[0], (x[1] / 2))).sortBy(lambda x: (-x[1], x[0])).collect()

    # write the initial betweeness as output 1
    with open(output_1, 'w+') as f:
        for elements in betweeness:
            edge = elements[0]
            value = elements[1]
            f.write(str(edge) + ',' + str(round(value, 5)) + '\n')

    # calculate degree
    degree = {}
    for node, connect in adj_dict.items():
        degree[node] = len(connect)


    # calculate number of edges
    m = len(edge_lst)

    # find adjacent matrix Adj (only store stored sorted edge pair as 1)
    Adj = {}
    for edge in edge_lst:
        Adj[edge] = 1

    # set the initial community modularity as -1 (smallest), stop when the Md start to decline

    final_community = [v_lst]
    modularity = calculate_modularity(final_community)


    # while loop, cut one or more edge at a time， terminate condition when modularity start to decrease

    while True:
        # find all edges with the highest betweenness, and remove them from the adj_dic
        bet_highest = betweeness[0][1]
        for bet in betweeness:
            if bet[1] == bet_highest:
                node1 = bet[0][0]
                node2 = bet[0][1]
                adj_dict[node1].remove(node2)
                adj_dict[node2].remove(node1)

        # After cut, find the temp communities that associated with the cut. Keep track of current community as final output
        current_comm = find_current_communities(v_lst)
        # print(len(current_comm))

        # calculate modularity of current community, compare it with the last loop, if decrease then stop. else continue
        current_mod = calculate_modularity(current_comm)
        # print(current_mod)

        # stop when current_mod <= max_modularity
        if current_mod >= modularity:
            modularity = current_mod
            final_community = current_comm
        else:
            break
        # print(current_mod)
        # update betweeness of the post-cut graph
        betweeness = g_rdd.map(lambda x: x[0]).distinct().map(lambda x: calculate_betweeness(x))\
            .flatMap(lambda x: [edge for edge in x]).reduceByKey(lambda x, y: x+y)\
            .map(lambda x: (x[0], (x[1] / 2))).sortBy(lambda x: (-x[1], x[0])).collect()

        print(betweeness[0])
    # print(modularity)
    # print(len(final_community))

    res = []
    for c in final_community:
        res.append(sorted(list(c)))
    res = sorted(res, key=lambda x: (len(x), x[0]))

    # write the final community as output 1
    with open(output_2, 'w+') as f:
        for a in res:
            f.write(str(a)[1:-1] + '\n')

    end = time.time()
    print(end - start)