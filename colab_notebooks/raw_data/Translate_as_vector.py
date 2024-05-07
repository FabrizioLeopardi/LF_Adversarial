f = open("NN_data2.txt","w")

f.write("NN_data2 = [")
with open("NN_data.txt","r") as Neural:
    for line in Neural:
        f.write(line.strip()+",\n")

f.write("]")
