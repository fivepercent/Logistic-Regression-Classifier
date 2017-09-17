class LogisticRegression():
    def __init__(self, tol, C, l_rate, max_iter):
        self.tol=tol if tol!=None else 0.0001
        self.C=C if C!=None else 1.0
        self.l_rate=l_rate if l_rate!=None else 1.0
        self.max_iter=max_iter if max_iter!=None else 100
    
    def fit(self, x, y):
        x=add_x0(x)
        n=x.shape[1]
        self.theta=np.zeros((n,1))
        for i in range(self.max_iter):
            fxs=fxValue(x, self.theta)
            dJ=(1.0/m)*(np.dot((fxs-y).T, x)+self.C*self.theta.T)
            self.theta-=self.l_rate*dJ.T
            if(np.linalg.norm(dJ)<self.tol):
                break

    def fxValue(self, x, theta):
        return 1.0/(1+np.exp(np.dot(x, theta)))
            
    def predict(self, test_sample):
        test_sample=self.add_x0(test_sample)
        return self.fxValue(test_sample, self.theta)
    
    def loadData(self, data):
        train_data = data.values
        m=train_data.shape[0]
        
        y=np.zeros((m,1))
        for i in range(m):
            if(train_data[:,train_data.shape[1]-1][i]=='是'):
                y[i]=1

        x=(np.float64)(train_data[:,0:train_data.shape[1]-1])
        return x,y
    
    def add_x0(self, x):
        return np.hstack((np.zeros((x.shape[0],1)) + 1, x))
    