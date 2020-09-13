l=input()
wc=[]
yc=[]
import collections
for i in range(len(l)):
    if l[i]!=":":
        if i+1<len(l) and l[i+1]==":":
            yc.append(l[i])
        else:
            wc.append(l[i])
n=int(input())
for _ in range(n):
    com=input().split()
    out = collections.OrderedDict()
    for i in range(len(com)):
        if "-" in com[i]:
            c=com[i][1]
            if c in wc:
                out[com[i]]=""
            if c in yc:
                out[com[i]]=com[i+1]
    out=(sorted(out.items(), key=lambda obj: obj[0]))
    ans=[]
    print(out)
    for k,v in out:
        ans.append(k)
        if v !="":ans.append(v)
    print(ans)
    print(f"Case {_+1}:"+" ".join(ans))