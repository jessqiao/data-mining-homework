from collections import defaultdict

# adj_dict = {'A': ['B', 'C'],
#             'B': ['A', 'D', 'E'],
#             'C': ['A', 'E'],
#             'D': ['B'],
#             'E': ['F', 'B', 'C'],
#             'F': ['E']}

adj_dict = {'E': ['D', 'F'],
            'D': ['E', 'F', 'B', 'G'],
            'F': ['D', 'E', 'G'],
            'B': ['D', 'A', 'C', 'K'],
            'G': ['D', 'F', 'K'],
            'A': ['B', 'C'],
            'C': ['A', 'B'],
            'K': ['B', 'G']}

sample = 'A'

parent_dic = defaultdict(list)
level_dic = defaultdict(list)
credit_dic = dict()
used_set = set()

root = sample

bfs_q = [root]
level = 0
level_dic[level] = root
credit_dic[root] = 1
used_set.add(root)
parent_dic[root] = []
next_level_q = []


while bfs_q or next_level_q:
    level += 1
    while bfs_q:
        de_q = bfs_q.pop(0)

        for children in adj_dict[de_q]:
            if children not in used_set:
                parent_dic[children].append(de_q)
                next_level_q.append(children)

        next_level_q = list(set(next_level_q))

    while next_level_q:
        a = next_level_q.pop(0)
        bfs_q.append(a)
        used_set.add(a)
        a_credit = 0
        for parent in parent_dic[a]:
            a_credit += credit_dic[parent]
        credit_dic[a] = a_credit
        level_dic[level].append(a)

print(level)
print(parent_dic)
print(level_dic)
print(credit_dic)

bet_dic = {}
edge_bet = {}

while level > 0:
    level -= 1
    for node in level_dic[level]:
        bet_dic[node] = bet_dic.get(node, 0) + 1
        for parent_node in parent_dic[node]:
            weight = credit_dic[parent_node] / credit_dic[node]
            edge_bet[tuple(sorted([node, parent_node]))] = bet_dic[node] * weight
            bet_dic[parent_node] = bet_dic.get(parent_node, 0) + bet_dic[node] * weight

print(bet_dic)
print(edge_bet)