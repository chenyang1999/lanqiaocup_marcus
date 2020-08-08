'''
问题描述
　　输入n个字符串，比较这些字符串的大小并统计并按字典序输出字符串及出现个数
输入格式
　　输入的第一行包含一个整数n，表示字符串个数。接下来n行，表示输入的字符串。
输出格式
　　输出n行，每行包含一个字符串及一个整数，表示字符串及出现个数。
样例输入
5
aab
bbc
aab
sdffg
sgjklsa
样例输出
aab 2
bbc 1
sdffg 1
sgjklsa 1
'''
n=int(input())
L=[]
for _ in range(n):
    s=input()
    L.append(s)
S=sorted(list(set(L)))
for s in S:
    print(s,L.count(s))
