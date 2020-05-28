import sys

def bifFix(infilename, outfilename):
    fin = open(infilename, "r")
    fout = open(outfilename, "w")
    lines = fin.readlines()
    for line in lines:
        if line.isspace():
            a = 1
        else:      
            if "variable" in line:
                parts = line.split('{')
                fout.write(parts[0]+"{\n")
                fout.write(parts[1]+"{")                
                endparts = parts[2].split("}")
                fout.write(endparts[0]+"}"+endparts[1]+"\n")
                fout.write("}\n")
            else:
                if '{' in line:
                    parts = line.split('{')
                    print(parts[0], "{")
                    print(parts[1])
                    fout.write(parts[0]+"{\n")
                    fout.write(parts[1])
                else:
                    print(line)
                    fout.write(line)
            
    fin.close()
    fout.flush()    
    fout.close() 

bifFix(sys.argv[1], sys.argv[2])

