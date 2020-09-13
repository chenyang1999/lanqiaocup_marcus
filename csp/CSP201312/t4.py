#题解:https://www.cnblogs.com/shenben/p/12270275.html
f=[[0,0,0,0,0,0]]
mod=int(1e9+7)
n=int(input())
for i in range(1,n+1):
    f.append([1])
    f[i].append((f[i-1][0]+f[i-1][1]*2)%mod)
    f[i].append((f[i-1][0]+f[i-1][2])%mod)
    f[i].append((f[i-1][1]+f[i-1][3]*2)%mod)
    f[i].append((f[i-1][1]+f[i-1][2]+f[i-1][4]*2)%mod)
    f[i].append((f[i-1][3]+f[i-1][4]+f[i-1][5]*2)%mod)
print(f[n][5])