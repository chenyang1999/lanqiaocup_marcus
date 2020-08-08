'''
问题描述
　　A同学的学习成绩十分不稳定，于是老师对他说：
    “只要你连续4天成绩有进步，那我就奖励给你一朵小红花。”
    可是这对于A同学太困难了。于是，老师对他放宽了要求：
    “只要你有4天成绩是递增的，我就奖励你一朵小红花。”
    即只要对于第i、j、k、l四天，满足i<j<k<l并且对于成绩wi<wj<wk<wl，
    那么就可以得到一朵小红花的奖励。现让你求出，A同学可以得到多少朵小红花。
输入格式
　　第一行一个整数n，表示总共有n天。第二行n个数，表示每天的成绩wi。
输出格式
　　一个数，表示总共可以得到多少朵小红花。
样例输入
6
1 3 2 3 4 5
样例输出
6
'''

n=int(input())
a=[int(x) for x in input().split()]
ans=0
# for i in range(len(a)):
#     for j in range(i+1,len(a)):
#         for k in range(j+1,len(a)):
#             for w in range(k+1,len(a)):
#                 if a[i]<a[j]<a[k]<a[w]:
#                     ans+=1
# print(ans)
dp=[[0 for i in range(4)]for j in range(n)]
for i in range(n):
    dp[i][0]=1

for i in range(n):
    for j in range(i+1,n):
        if a[i]<a[j]:
            dp[j][1]+=dp[i][0]
            dp[j][2]+=dp[i][1]
            dp[j][3]+=dp[i][2]
for i in range(n):
    ans+=dp[i][3]
print(ans)