'''
问题描述
　　集合M至少有两个元素（实数），且M中任意两个元素差的绝对值都大于2，则称M为“翔集合”，已知集合S={1,2...,n},请求出n的子集中共有多少个翔集合。
输入格式
　　输入共一行，一个整数n.(n>=2)
输出格式
　　输出共一行，一个整数表示S的子集中共有多少个翔集合，由于个数可能过大，请输出这个值除以1000007的余数。
样例输入
4
样例输出
1
'''

n=int(input())
ans=1
for i in range(1,n-3):
    ans=ans * i % 1000007
print(ans)
