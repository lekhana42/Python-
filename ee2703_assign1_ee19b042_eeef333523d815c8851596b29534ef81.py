from sys import argv, exit
if len(argv) != 2:
        print('\nUsage: %s <inputfile>' % argv[0])
        exit()
CIRCUIT = '.circuit'
END = '.end'
try:
    with open(argv[1]) as f:
        lines = f.readlines()
        start = -1; end = -2
        for line in lines:              # extracting circuit definition start and end lines
            if CIRCUIT == line[:len(CIRCUIT)]:
                start = lines.index(line)
            elif END == line[:len(END)]:
                end = lines.index(line)
                break
        if start >= end:                # validating circuit block
            print('Invalid circuit definition')
            exit(0)
        a=[]
        for eachline in lines[start+1:end]:
            a.append(eachline.split('#')[0].split())
        a.reverse()
        for i in a:
            i.reverse()
        for i in a:
            print(" ".join(i))
except IOError:
    print('Invalid file')
    exit()

