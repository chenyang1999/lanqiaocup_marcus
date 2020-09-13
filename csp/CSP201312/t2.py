l=input().split("-")
a=l[-1]
b=0
num="".join(l[:-1])
for i,n in enumerate(num):
    b+=((i+1)*int(n))
b%=11
b=str(b)
if b=='10':b='X'
# print(b,a)
if b==a:
    print("Right")
else:
    print("-".join(l[:-1]+[b]))
'''
样例输入
0-670-82162-4
样例输出
Right
样例输入
0-670-82162-0
样例输出
0-670-82162-4
'''