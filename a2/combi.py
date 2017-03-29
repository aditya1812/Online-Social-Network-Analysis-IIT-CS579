from itertools import combinations

feat = [77, 88 , 99]
punct = [True, False]
freq = [2,5,10]

comb = []
for i in range(1, len(feat) + 1):
    for c in combinations(feat, i):
        comb.append(list(c))

print(comb)
x = []
for c in comb:
    for y in punct:
        for z in freq:
            x.append([c,y,z])
print(x)

for i in x:
    if True in i:
        print(i[0])
