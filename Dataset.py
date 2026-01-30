class Dataset :
    def __init__(self,X,y):
        self.X = X
        self.y = y 
        #ToDo add weights 


class State : 
    def __init__(self,populations,hof,cycles_remain):
        self.populations = populations
        self.hof = hof 
        self.cycles_remain = cycles_remain
        