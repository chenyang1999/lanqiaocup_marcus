'''
问题描述
　　有一条长为n的走廊，小明站在走廊的一端，每次可以跳过不超过p格，每格都有一个权值wi。
　　小明要从一端跳到另一端，不能回跳，正好跳t次，请问他跳过的方格的权值和最大是多少？
输入格式
　　输入的第一行包含两个整数n, p, t，表示走廊的长度，小明每次跳跃的最长距离和小明跳的次数。
　　接下来n个整数，表示走廊每个位置的权值。
输出格式
　　输出一个整数。表示小明跳过的方格的权值和的最大值。
样例输入
8 5 3
3 4 -1 -100 1 8 7 6
样例输出
12
'''

n,p,t=[int(x) for x in input().split()]
a=[int(x) for x in input().split()]
a=[0]+a
dp=[[0 for i in range(t+1)]for j in range(n+1)]
inf=2**32
for i in range(n-p+1,n+1):
    dp[i][1]=a[i]

for i in range(n-1,-1,-1):
    for j in range(2,t+1):
        dp[i][j]=-inf
        for k in range(i+1,min(i+p+1,n+1)):
            dp[i][j]=max(dp[i][j],dp[k][j-1])
        if(dp[i][j]!=-inf):
            dp[i][j]=dp[i][j]+a[i]

print(dp[0][t])