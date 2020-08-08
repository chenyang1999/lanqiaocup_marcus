'''
输出格式
输出杨辉三角形的前n行。每一行从这一行的第一个数开始依次输出，中间使用一个空格分隔。请不要在前面输出多余的空格。
样例输入
4
样例输出
1
1 1
1 2 1
1 3 3 1
'''
n=int(input())
L=[[0 for i in range(n+1)] for j in range(n+1)]
# print(L)
L[0][0]=1
for i in range(1,n+1):
    for j in range(1,i+1):
        print(L[i-1][j-1]+L[i-1][j],end=" ")
        L[i][j]=L[i-1][j-1]+L[i-1][j]
    print()
'''
Python3中初始化一个多维数组，通过for range方法。以初始化二维数组举例：

>>> test = [[ 0 for i in range(2)] for j in range(3)]
>>> test
[[0, 0], [0, 0], [0, 0]]
1
2
3
初始一个一维数组，可以使用*

>>> test = [ 0 for i in range(3)]
>>> test
[0, 0, 0]
'''