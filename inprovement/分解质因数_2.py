'''
问题描述
　　给定一个正整数n，尝试对其分解质因数
输入格式
　　仅一行，一个正整数，表示待分解的质因数
输出格式
　　仅一行，从小到大依次输出其质因数，相邻的数用空格隔开
'''
n=int(input())
L=list(range(2,int(n**0.5)+1))
Z=[]
# print(L)
for zs in L:
    if zs:
        for hs in range(zs+zs,int(n**0.5),zs):
            # print(hs)
            L[hs-2]=0
        Z.append(zs)

# print(L)
# print(Z)
ans=[]
for zs in Z:
    while n%zs==0:
        ans.append(str(zs))
        n=n//zs
        # print(n)
    if n==1:
        break

if n!=1:ans.append(str(int(n)))
print(" ".join(ans))