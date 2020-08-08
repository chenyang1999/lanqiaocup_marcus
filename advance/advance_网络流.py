'''
问题描述
　　一个有向图，求1到N的最大流
输入格式
　　第一行N M，表示点数与边数
　　接下来M行每行s t c表示一条从s到t的容量为c的边
输出格式
　　一个数最大流量
样例输入
6 10
1 2 4
1 3 8
2 3 4
2 4 4
2 5 1
3 4 2
3 5 2
4 6 7
5 4 6
5 6 3
样例输出
8
'''

# pass 我还不会做网络流的题目,先马着

# Python program for implementation of Ford Fulkerson algorithm
import copy
from collections import deque


def hasPath(Gf, s, t, path):
    # BFS algorithm
    V = len(Gf)
    visited = list(range(V))
    for i in range(V):
        visited[i] = False
    visited[s] = True
    queue = deque([s])
    while queue:
        temp = queue.popleft()
        if temp == t:
            return True
        # print("temp =", temp)
        for i in range(V):
            if not visited[i] and (Gf[temp][i] > 0):
                queue.append(i)
                visited[i] = True
                path[i] = temp   # record precursor
    return visited[t]


def max_flow(graph, s, t):
    maxFlow = 0
    Gf = copy.deepcopy(graph)
    V = len(Gf)
    path = list(range(V))
    while hasPath(Gf, s, t, path):
        min_flow = float('inf')

        # find cf(p)
        v = t
        while v != s:
            u = path[v]
            min_flow = min(min_flow, Gf[u][v])
            v = path[v]
        # print(min_flow)

        # add flow in every edge of the augument path
        v = t
        while v != s:
            u = path[v]
            Gf[u][v] -= min_flow
            Gf[v][u] += min_flow
            v = path[v]

        maxFlow += min_flow
    return maxFlow

M=0
n,m=[int(x) for x in input().split()]
capacity = [[M for i in range(n)]for j in range(n)]

for _ in range(m):
    a,b,c=[int(x) for x in input().split()]
    capacity[a-1][b-1]=c
flow = max_flow(capacity, 0,n-1)
# print("flow =", flow)
print(flow)