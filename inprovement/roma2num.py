def roma2num(s):
    ROMAN={
        'I':1,
        'V':5,
        'X':10,
        'L':50,
        'C':100,
        'D':500,
        'M':1000
    }
    if s=="":return 0
    index = len(s)-2;
    sum=ROMAN[s[index]]
    while index>=0:
        if (ROMAN[s[index]])<ROMAN[s[index+1]]:
            sum-=ROMAN[s[index]]
        else:
            sum+=ROMAN[s[index]]
        index -= 1
        return sum

