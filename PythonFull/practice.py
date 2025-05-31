def findDuplicated(a):
    result=[]
    for j in range(len(a)):
        first=a[j]
        for i in range(1,len(a)):
            if first==a[i]:
                result.append(first)

        

    
                
print(findDuplicated([4, 3, 2, 7, 8, 2, 3, 1]))