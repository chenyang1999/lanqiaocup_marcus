'''
问题描述
　　汉诺塔是一个古老的数学问题：
　　有三根杆子A，B，C。A杆上有N个(N>1)穿孔圆盘，盘的尺寸由下到上依次变小。要求按下列规则将所有圆盘移至C杆：
　　每次只能移动一个圆盘；
　　大盘不能叠在小盘上面。
　　提示：可将圆盘临时置于B杆，也可将从A杆移出的圆盘重新移回A杆，但都必须遵循上述两条规则。

　　问：如何移？最少要移动多少次？
输入格式
　　一行，包含2个正整数，一个是N（N<=15)，表示要移动的盘子数；一个是M，表示在最少移动d第M步
输出格式
　　共2行。
　　第一行输出格式为：#No: a->b，表示第M步骤具体移动方法，其中No表示第M步移动的盘子的编号（N个盘子从上到下依次编号为1到n），表示第M步是将No号盘子从a杆移动到b杆（a和b的取值均为{A、B、C}）。
　　第2行输出一个整数，表示最少移动步数。
'''

n,m=[int(x)for x in input().split()]
counts = 0
def move(n,a,b,c):
    global counts
    if n==1:
        counts=counts+1
        if m==counts:
            print("#1: ",a,'->',c,sep="")
        return
    else:
        move(n-1,a,c,b)
        counts = counts + 1
        if m == counts :
            print("#", m, ": ", a, '->', c, sep="")
        move(n-1,b,a,c)
move(n,"A","B","C")
print(2**n-1)