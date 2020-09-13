# 蓝桥杯-python刷题记录

`Marcus扬`

刷题网站:https://loj.ac/

算法学习网站:https://oi-wiki.org/

python 数据集算法模板库:https://github.com/edisonleolhl/DataStructure-Algorithm

i 了 i 了

## tips

### python 不同进制的转换

需要注意,巧用十进制作为桥梁

```python
## 十六进制 到 十进制
int('0Xf',16) 
## 八进制转 到 十进制
int('20',8)
## 二进制转 到 十进制
int('10101',2)


## 十进制 转 十六进制
>>> hex(1033)
'0x409'
## 二进制 转 十六进制
## 就是二进制先转成十进制，再转成十六进制。
>>> hex(int('101010',2))
'0x2a'
## 八进制到 十六进制
##就是 八进制先转成 十进制， 再转成 十六进制。
>>> hex(int('17',8))
'0xf'

## 十进制装二进制
bin(10,2)
## 十进制转八进制
oct(10,8)
```

### python 二维数组初始化

参考博客:[python3 初始二维数组](https://blog.csdn.net/qq_24504591/article/details/88222491?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.channel_param)

Python3中初始化一个多维数组，通过`for range`方法。以初始化二维数组举例：

```python
arr = [[] for i in range(5)]
>>> [[], [], [], [], []]
arr = [[0, 0] for i in range(5)]
arr[2].append(2)
>>> [[0, 0], [0, 0], [0, 0, 2], [0, 0], [0, 0]]
12345
```

初始一个一维数组，可以使用*或者`for range`

```python
arr1 = [None for i in range(5)]
>>> [None, None, None, None, None]
arr2 = [None]*5
>>> [None, None, None, None, None]
1234
```

但是用*初始化二维数组则会在修改数组内容时出现错误，例如：

```python
arr = [[0, 0]]*5
arr[2] = 2
>>> [[0, 0], [0, 0], 2, [0, 0], [0, 0]] # 直接复制不会出现错误
arr[2].append(2)
>>> [[0, 0, 2], [0, 0, 2], [0, 0, 2], [0, 0, 2], [0, 0, 2]]
arr[2][1] = 5
>>> [[0, 5], [0, 5], [0, 5], [0, 5], [0, 5]]
1234567
```

而使用`for range`初始化不会产生该问题，range会另外开辟一个新的内存地址；*会指向同一个内存地址，改变值会其内存地址指向的值，从而改变所有的值。

### python 保留小数

```python
round(x,2)
#对 x 保留 2 位小数
注意可能有的 bug,就是会出现丢弃末尾的 0 的可能,这个时候只能够在打印的时候使用控制字符.
```

### python 网络流

参考网站:https://www.jianshu.com/p/efb2d79e2b0f

geeks:https://www.geeksforgeeks.org/ford-fulkerson-algorithm-for-maximum-flow-problem/?ref=lbp



### Python Euler's Totient Function 

```python
# Python3 program to calculate  
# Euler's Totient Function 
def phi(n): 
      
    # Initialize result as n 
    result = n;  
  
    # Consider all prime factors 
    # of n and subtract their 
    # multiples from result 
    p = 2;  
    while(p * p <= n): 
          
        # Check if p is a  
        # prime factor. 
        if (n % p == 0):  
              
            # If yes, then  
            # update n and result 
            while (n % p == 0): 
                n = int(n / p); 
            result -= int(result / p); 
        p += 1; 
  
    # If n has a prime factor 
    # greater than sqrt(n) 
    # (There can be at-most  
    # one such prime factor) 
    if (n > 1): 
        result -= int(result / n); 
    return result; 
```



## 矩阵快速幂

翔集合:https://blog.csdn.net/rwrsgg/article/details/106185675

```c++
//矩阵快速幂实现翔集合 
#include <iostream>
#include <algorithm>
#include <cstring>
using namespace std;
typedef long long ll;
struct node{
	ll A[5][5];
	node(){
		for(int i = 0;i<5;i++)
		 for(int j = 0;j<5;j++)
		  A[i][j]=0;
		}
}x,y;
ll n;
void set()
{
	x.A[0][0]=x.A[0][2]=x.A[0][3]=1;
	x.A[1][0] = 1;
	x.A[2][1] =1;
	x.A[3][3]=x.A[3][4]=1;
	x.A[4][4]=1;
	
    y.A [3][0] =1;
	y.A [4][0] = 1;
}
struct node Mul(node tmp1,node tmp2)
{
	node tmp3;
	for(int i =0;i<5;i++)
	{
		for(int j = 0;j<5;j++)
		{
			for(int k = 0;k<5;k++)
			{
				tmp3.A[i][j]+=(tmp1.A[i][k]*tmp2.A[k][j])%1000007;
			}
		}	
    }
    return tmp3;
}
struct node quick2_pow(ll k)
{
	node ans = x;
	//cout<<k<<endl;
	while(k)
	{  
		if(k&1) ans=Mul(ans,x);
		x = Mul(x,x);
	    k>>=1;
	}
  return ans;
}

int main()
{   
	set();
	cin>>n;
	if(n<4) 
	{
		printf("0\n");
		return 0;
	}
	node s;
	s = Mul(quick2_pow(n-4),y);

	printf("%lld\n",s.A[0][0]%1000007);
} 

```



----

## TO-DO-LIST

1. 学习 python 的 网络流算法的实现
2. 