'''
输入格式
　　输入一行，包含一个正整数N。
输出格式
　　共两行。
　　第1行包含一个整数，表示N以内质数的个数。
　　第2行包含若干个素数，每两个素数之间用一个空格隔开，素数从小到大输出。
'''
n=int(input())
L=list(range(n+1))
L[0]=0
L[1]=0
Z=[]
# print(L)
for zs in L:
    if zs:
        for hs in range(zs+zs,n+1,zs):
            # print(hs)
            L[hs]=0
        Z.append(str(zs))
print(len(Z))
print(" ".join(Z))