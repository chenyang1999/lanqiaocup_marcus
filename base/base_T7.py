'''
问题描述
利用字母可以组成一些美丽的图形，下面给出了一个例子：
ABCDEFG
BABCDEF
CBABCDE
DCBABCD
EDCBABC
这是一个5行7列的图形，请找出这个图形的规律，并输出一个n行m列的图形。

输入格式
输入一行，包含两个整数n和m，分别表示你要输出的图形的行数的列数。
输出格式
输出n行，每个m个字符，为你的图形。

思路:每行都是对称的,对称中心是 A,
每一行 A 向右移一格
'''
n,m=input().split()
n=int(n)
m=int(m)
alpha=[chr(i) for i in range(65+25,65,-1)]+[chr(i) for i in range(65,65+26)]
for i in range(n):
    print("".join(alpha[26-i-1:26-i+m-1]))