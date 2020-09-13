n=int(input())
l=[int(x) for x in input().split()]
dic={}
for i in l:
    if i not in dic.keys():
        dic[i]=1
    else:
        dic[i]+=1
ans=l[0]
for i in dic.keys():
    if dic[i]>dic[ans]:ans = i
print(ans)
'''
样例输入
6
10 1 10 20 30 20
样例输出
10
'''
