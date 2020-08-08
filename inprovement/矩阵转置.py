'''
【问题描述】

　　给定一个n×m矩阵相乘，求它的转置。其中1≤n≤20，1≤m≤20，矩阵中的每个元素都在整数类型（4字节）的表示范围内。
【输入格式】
　　第一行两个整数n和m；
　　第二行起，每行m个整数，共n行，表示n×m的矩阵。数据之间都用一个空格分隔。
【输出格式】

　　共m行，每行n个整数，数据间用一个空格分隔，表示转置后的矩阵。
'''
n,m=[int(x) for x in input().split()]
L=[]

for _ in range(n):
    L.append(input().split())

for i in range(m):
    for j in range(n):
        print(L[j][i],end=" ")
    print()