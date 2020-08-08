'''
问题描述
　　输入一个小写字母，转换成大写字母。
输入格式
　　输入小写字母
输出格式
　　输出转换后的大写字母
'''
s=input()
L=[]
for c in s:
    if c.islower():
        L.append(c.upper())
    else:
        L.append(c.lower())
print("".join(L))