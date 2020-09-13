L=[]
with open('zht.txt',"r") as f:
	for line in f:
		L.append(line.split())
for line in L:
	print(line)
sum_Weighted_credits=0
avg=0
sum_credits=0
for line in L:
	if (line[4]=='1'):
		sum_Weighted_credits+=float(line[0])*float(line[1])
		sum_credits+=float(line[0])
	if (line[4]=='-1'):
		sum_Weighted_credits+=60*float(line[0])
		sum_credits+=float(line[0])
	print(line,"avg",sum_Weighted_credits/sum_credits)
print("sum_Weighted_credits =",sum_Weighted_credits)
print("sum_credits =",sum_credits)
print("avg",sum_Weighted_credits/sum_credits)