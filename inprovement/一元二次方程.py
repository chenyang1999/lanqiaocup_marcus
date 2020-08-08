'''
问题描述
　　编写一个求解一元二次方程的实数根的程序，方程的系数由用户在运行xh
输入格式
　　输入一行三个整数分别为一元二次方程的三个系数，数据之间以空格隔开
输出格式
　　输出一行为方程的实数根（若两根不同 较大的在前 两根以空格隔开 若两根相同 输出一个 若无根 输出 NO ）
'''
a,b,c=[int(x) for x in input().split()]
delta=(b**2-4*a*c)
# print(delta)
if delta<-0.0000000001:
    print('NO')
elif abs(delta-0)<0.0000000001:
    print(round(-b/(2*a)))
else:
    print(round(max((-b-delta**0.5)/(2*a),(-b+delta**0.5)/(2*a))),round(min((-b-delta**0.5)/(2*a),(-b+delta**0.5)/(2*a))))
