'''
问题描述
小明为某机构设计了一个十字型的徽标（并非红十字会啊），如下所示：

..$$$$$$$$$$$$$..
..$...........$..
$$$.$$$$$$$$$.$$$
$...$.......$...$
$.$$$.$$$$$.$$$.$
$.$...$...$...$.$
$.$.$$$.$.$$$.$.$
$.$.$...$...$.$.$
$.$.$.$$$$$.$.$.$
$.$.$...$...$.$.$
$.$.$$$.$.$$$.$.$
$.$...$...$...$.$
$.$$$.$$$$$.$$$.$
$...$.......$...$
$$$.$$$$$$$$$.$$$
..$...........$..
..$$$$$$$$$$$$$..
对方同时也需要在电脑dos窗口中以字符的形式输出该标志，并能任意控制层数。

输入格式
一个正整数 n (n<30) 表示要求打印图形的层数。
输出格式
对应包围层数的该标志。
'''

def p(L:list):
    for i in L:
        for j in i:
            print(j,sep="",end="")
        print()
        
n=int(input())
m=5+n*4
L=[['.' for i in range(m)] for j in range(m)]

def dg(L: list,bj):
    if bj==n+1:
        return L
    start=bj*2+2
    end=m-bj*2-2
    # print(start,end,m)
    for i in range(start,end):
        L[start-2][i]="$"
    for i in range(start,end):
        L[end+1][i]="$"
    for i in range(start,end):
        L[i][start-2]="$"
    for i in range(start,end):
        L[i][end+1]="$"
    # p(L)
    # print("-------------------------------")
    return dg(L,bj+1)
dg(L,0)

def dg2(L: list,bj):
    if bj==n+1:
        return L
    start=bj*2+2
    end=m-bj*2-3
    # print(start,end,m)
    L[start][start]="$"
    L[start-1][start]="$"
    L[start][start-1]="$"

    L[start][end]="$"
    L[start][end+1]="$"
    L[start-1][end]="$"

    L[end][end]="$"
    L[end+1][end]="$"
    L[end][end+1]="$"

    L[end][start]="$"
    L[end][start-1]="$"
    L[end+1][start]="$"

    # p(L)
    # print("-------------------------------")
    return dg2(L,bj+1)
dg2(L,0)
p(L)