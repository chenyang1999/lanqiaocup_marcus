n=int(input())
l=[int(x) for x in input().split()]
area=0
for i,x in enumerate(l):
    hight=x
    for j in range(i,len(l)):
        hight=min(hight,l[j])
        area=max(((j-i+1)*hight),area)
print(area)