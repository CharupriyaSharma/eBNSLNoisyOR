from numpy import genfromtxt
from CPDClass import CPD

class Data:

    # Returns the domain of a node by checking all values it takes from a numpy array 
    def domainOf(self, node):
        domain = []
        for row in self.data:
            if row[node] not in domain:
                domain.append(row[node])
        return domain    

    def domainsOf(self):
        domains = []
        for node in range(0,len(self.data[0])):
            domains.append(self.domainOf(node))
        return domains  
    
    # Reads integer data from a CSV file with delimiter ','.
    def readFromCSVFile(self, filename):
        self.data = genfromtxt(filename, delimiter=',', dtype=int)    
        self.domains = self.domainsOf()
        self.numberOfColumns = len(self.data[0])
        self.numberOfRows = len(self.data)
        
    def computeCPD(self, childNode, parentSet):
        result = CPD(self, childNode, parentSet)        
        return result  
