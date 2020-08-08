'''
问题描述
100 可以表示为带分数的形式：100 = 3 + 69258 / 714。
还可以表示为：100 = 82 + 3546 / 197。
注意特征：带分数中，数字1~9分别出现且只出现一次（不包含0）。
类似这样的带分数，100 有 11 种表示法。
输入格式
从标准输入读入一个正整数N (N<1000*1000)
输出格式
程序输出该数字用数码1~9不重复不遗漏地组成带分数表示的全部种数。
注意：不要求输出每个表示，只统计有多少表示法！
'''

'''
n=int(input())
# n=100
num=list(range(1,10))
counter = 0
s=list(range(1,10))
vs=[1]*10

def dfs(a,b,c,la,lb,lc):
    global counter
    if (la==0 and lb==0 and lc==0):
        # print(a,b,c)
        if (b%c==0 and  b/c==n-a):
            counter= counter + 1
            # print(a,b,c)
            return
    for i in s:
        if vs[i]==1:
            #
            vs[i]=0
            if (la>0):
                dfs(a*10+i,b,c,la-1,lb,lc)
            else:
                if lb>0:
                    dfs(a,b*10+i,c,la,lb-1,lc)
                else:
                    if (lc>0):
                        dfs(a,b,c*10+i,la,lb,lc-1)
            vs[i]=1

for i in range(1,8):
    for j in range(1,8-i):
        if (9-i-j>= 1):
            dfs(0, 0, 0, i, j, 9-i-j)
print(counter)
'''
from itertools import permutations

n = int(input())
# n=100
num = list(range(1, 10))
count = 0

for s in permutations(num):
    # print(s)
    s = "".join([str(x) for x in s])
    for i in range(1, 8):
        for j in range(i + 1, 9):
            if int(s[:i]) + int(s[i:j]) / int(s[j:]) == n:
                # print(s[:i],s[i:j],s[j:])
                count += 1
print(count)
