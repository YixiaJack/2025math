
col=[[] for _ in range(10)]
with open(filename,'r') as f:
    for lines in f:
        numbers=lines.strip().split()
        for i in range(10):
            col[i].append(int(numbers[i]))
for co
