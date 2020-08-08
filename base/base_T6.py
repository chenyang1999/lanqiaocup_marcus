'''
问题描述
对于长度为5位的一个01串，每一位都可能是0或1，一共有32种可能。它们的前几个是：
00000
00001
00010
00011
00100
请按从小到大的顺序输出这32种01串。
'''
for i in range(2**5):
    print("0"*(5-len(bin(i)[2:]))+bin(i)[2:])