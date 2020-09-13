n=int(input())
l=[int(x) for x in input().split()]
ans=0
for i in l:
    if (-i) in l:ans+=1
print(ans//2)