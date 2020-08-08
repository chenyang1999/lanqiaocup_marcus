'''
问题描述
　　计算1*(1+k)*(1+2*k)*(1+3*k)*...*(1+n*k-k)的末尾有多少个0，最后一位非0位是多少。
输入格式
　　输入的第一行包含两个整数n, k。
输出格式
　　输出两行，每行一个整数，分别表示末尾0的个数和最后一个非0位。
样例输入
15 2
样例输出
0
5
'''
n,k = [int(x) for x in input().split()]
counter=0
ans=1
for i in range(1,n):
    ans=ans*(1+i*k)
    while ans%10==0:
        ans=ans//10
        counter=counter+1
    ans=ans%100000
print(counter)
print(ans%10)