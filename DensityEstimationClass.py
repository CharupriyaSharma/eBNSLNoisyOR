from BIFClass import BIFFile
            
class DensityEstimation:

    def __init__(self, biffilename, observationfilename, debug = False):
        self.bif = BIFFile(biffilename)
        obsfile = open(observationfilename, "r")
        lines = obsfile.readlines()             
        for line in lines[1:]:
            line = line.strip()
            tokens = line.split(",")
            product = 1
            for variable in self.bif.variables.values():                
                key = tokens[variable.index]
                for parent in variable.parents:
                    key = key + tokens[parent.index]
                if debug:    
                    print("Debug", variable.name, 
                          ", index=", variable.index, 
                          ", parents", [p.name for p in variable.parents],
                          ", obs=", tokens[variable.index], 
                          ", parentObs=", key)
                    print("Debug Prob", float(variable.prob[key]))
                product = product * float(variable.prob[key])
            print(product)    
                
                

#class DensityEstimation:
    
test = BIFFile("asia.bif")
test.printVariables()
de = DensityEstimation("asia.bif", "100.csv")
