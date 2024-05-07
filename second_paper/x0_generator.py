g = open("x_0.ppm","w")
q = open("data/x_0_data.txt","w")

g.write("P3\n")
g.write("28 28\n")
g.write("255\n")

scale_factor = 0.3844543695449829 # This value is the result of numerical_attack.py

with open("data/x_seed_data.txt","r") as f:
    for line in f:
        e = float(line)
        x = int(e*255.0*scale_factor)
        g.write(str(round(x))+" "+str(round(x))+" "+str(round(x))+"\n")
        q.write(str(float(x/255.0))+"\n")
        

g.close()
q.close()
