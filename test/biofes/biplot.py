import pandas
import numpy
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt
from scipy import stats

################################################################################
############################# functions ########################################
################################################################################

def standardize(data,meth=None):
    global data_stan
    if meth == None:
        data_stan = data
    if meth == 1:
        medias = data.mean(axis=0)
        desv = data.std(axis=0)
        data_stan = (data-medias)/desv
    return data_stan

def SVD(M,dimen,niter=5,state=0):
    U, Sigma, VT = randomized_svd(M, n_components=dimen,n_iter=niter,random_state=state)
    return U, Sigma, VT

#def Inertia(M,dimen,niter=5,state=None):
#    U, Sigma, VT = SVD(M,dimen,niter,state)
#    EV = numpy.power(Sigma,2)
#    Inert = EV/numpy.sum(EV) * 100
#    return EV, Inert

#def Contributions(M,dimen,n,p,niter=5,state=None):
#    U, Sigma, VT = SVD(M,dimen,niter,state)

#    R = U.dot(numpy.diag(Sigma[:dimen]))
#    C = numpy.transpose(VT).dot(numpy.diag(Sigma[:dimen]))

#    sf = numpy.sum(numpy.power(M,2),axis=1)
#    cf = numpy.zeros((n,dimen))
#    for k in range(0,dimen):
#        cf[:,k] = numpy.power(R[:,k],2)*100/sf

#   sc = numpy.sum(numpy.power(M,2),axis=0)
#   cc = numpy.zeros((p,dimen))
#   for k in range(0,dimen):
#       cc[:,k] = numpy.power(C[:,k],2)*100/sc

#   return cf, cc

def Factor2Binary(y,Name = None):
    if Name == None:
        Name = "C"
    ncat = len(list(set(y)))
    n = len(y)
    Z = pandas.DataFrame(0, index=numpy.arange(len(y)), columns=list(set(y)))
    for col in Z.columns:
        for i in range (0,n):
            if y[i] == col:
                Z[col].iloc[i] = 1
    return Z

def matrixsqrt(M,dim,tol=numpy.finfo(float).eps,inv=True):
    U, Sigma, VT = randomized_svd(M, n_components=dim, n_iter=5, random_state=None)
    nz = Sigma > tol
    if inv==True:
        S12 = U.dot(numpy.diag(1/numpy.sqrt(Sigma[nz]))).dot(VT[nz,:])
    else:
        S12 = U.dot(numpy.diag(numpy.sqrt(Sigma[nz]))).dot(VT[nz,:])
    return S12

################################################################################
############################# Classical biplot #################################
################################################################################

class Classic(object):
    '''
    Gabriel biplots
    '''

    def __init__(self, data, dim, alpha = 1, method=None,niter=5,state=None):
        if isinstance(data , pandas.core.frame.DataFrame):
            self.col_names = list(data.columns)
            self.data = data.values
        elif isinstance(data , numpy.ndarray):
            self.col_names = ['Var_'+str(i+1) for i in range(data.shape[1])]
            self.data = data
        else:
            raise ValueError('not pandas DataFrame nor numpy ndarray')

        if isinstance(dim, (int, float)):
            self.dim = dim
        else:
            raise ValueError('not numeric')

        if self.dim > self.data.shape[1]:
            raise ValueError('dim bigger than p')

        if (alpha>=0 and alpha<=1):
            self.alpha = alpha
        else:
            raise ValueError('not between 0 and 1')
            
        self.data_st = standardize(self.data,meth=method)
        n, p = self.data_st.shape
        
        #SVD
        U, Sigma, VT = SVD(self.data_st,self.dim,niter,state)
        
        # EV / INERTIA
        self.EV = numpy.power(Sigma,2)
        self.Inert = self.EV/numpy.sum(self.EV) * 100
        
        # CONTRIBUTIONS
        
        R = U.dot(numpy.diag(Sigma[:self.dim]))
        C = numpy.transpose(VT).dot(numpy.diag(Sigma[:self.dim]))
        
        sf = numpy.sum(numpy.power(self.data_st,2),axis=1)
        cf = numpy.zeros((n,self.dim))
        for k in range(0,self.dim):
            cf[:,k] = numpy.power(R[:,k],2)*100/sf
        
        sc = numpy.sum(numpy.power(self.data_st, 2),axis=0)
        cc = numpy.zeros((p,self.dim))
        for k in range(0,self.dim):
            cc[:,k] = numpy.power(C[:,k],2)*100/sc
        
        self.RowCont, self.ColCont = cf, cc

        R = R.dot(numpy.diag(numpy.power(Sigma,self.alpha)))
        C = C.dot(numpy.diag(numpy.power(Sigma,1-self.alpha)))

        sca = numpy.sum(numpy.power(R,2))/self.data.shape[0]
        scb = numpy.sum(numpy.power(C,2))/self.data.shape[1]
        scf = numpy.sqrt(numpy.sqrt(scb/sca))

        self.RowCoord = R*scf
        self.ColCoord = C/scf
    
    def plot(self, target = None, dim = [0, 1] , xlim=[-2,2] , ylim= [-2,2], figx = 10, figy = 10, arrow_width = 0.1, font_size = 12):
        if isinstance(target, (numpy.ndarray, list, pandas.core.series.Series)):
            fig = plt.figure(figsize=(figx,figy))
            ax1 = fig.add_subplot(111)

            ax1.scatter(self.RowCoord[:, dim[0]], self.RowCoord[:, dim[1]], c = target)
            for i in range(0,self.ColCoord.shape[0]):
                ax1.arrow(0 ,0 , self.ColCoord[i, dim[0]],self.ColCoord[i, dim[1]], width = arrow_width )
                ax1.text(self.ColCoord[i ,dim[0]], self.ColCoord[i,dim[1]] , self.col_names[i], fontsize=font_size)

        plt.show()
        
class Canonical(object):
    '''
    Canonical selflot (Vicente-Villardon)
    '''

    def __init__(self, data,dim,GroupNames,y, method=None,niter=5,state=0):
        if isinstance(data , pandas.core.frame.DataFrame):
            self.col_names = list(data.columns)
            self.data = data.values
        elif isinstance(data , numpy.ndarray):
            self.col_names = ['Var_'+str(i+1) for i in range(data.shape[1])]
            self.data = data
        else:
            raise ValueError('not pandas DataFrame nor numpy ndarray')

        if isinstance(dim, (int, float)):
            self.dim = dim
        else:
            raise ValueError('not numeric')
        if self.dim>self.data.shape[1]:
            raise ValueError('dim bigger than p')
        if isinstance(GroupNames, (list)):
            self.GroupNames = GroupNames
        else:
            raise ValueError('not numeric')

        if isinstance(y, list):
            self.target = y
        else:
            raise ValueError('not list')
            
        if len(y) != data.shape[0]:
            raise ValueError( 'data ({}) and y ({}) have different length'.format(data.shape[0],len(y)) )

        self.data_st = standardize(self.data,meth=method)
        data_std = self.data_st
        g = len(self.GroupNames)
        n = self.data.shape[0]
        m = self.data.shape[1]
        r = numpy.min(numpy.array([g - 1, m]))

        #Groups to Binary
        Z = Factor2Binary(self.target)
        ng = Z.sum(axis=0)
        S11 = (Z.T).dot(Z).values
        #print('S11 : {} / Z.T : {} / data_std : {}'.format(S11.shape, Z.T.shape, data_std.shape))
        Xb = numpy.linalg.inv(S11).dot(Z.T).dot(data_std)
        B = (Xb.T).dot(S11).dot(Xb)
        S = (data_std.T).dot(data_std) - B
        Y = numpy.power(S11,0.5).dot(Xb).dot(matrixsqrt(S,self.dim,inv=True))

        U, Sigma, VT = SVD(Y,self.dim,niter,state)

        #Variable_Coord
        H = matrixsqrt(S,self.dim,inv=False).dot(numpy.transpose(VT[0:r,:]))
        self.Var_Coord = H
        #Canonical_Weights
        B = matrixsqrt(S,self.dim,inv=True).dot(numpy.transpose(VT[0:r,:]))
        self.Can_Weights = B
        #Group_Coord
        J = Xb.dot(B)
        self.Group_Coord = J
        #Individual_Coord
        V = data_std.dot(B)
        self.Ind_Coord = V

        sct = numpy.diag((V.T).dot(V))
        sce = numpy.diag((J.T).dot(S11).dot(J))
        scr = sct -sce
        fs = (sce/(g - 1))/(scr/(n - g))

        #eigenvectors
        vprop = Sigma[:r]
        self.vprop = vprop
        #Inertia
        iner = (numpy.power(vprop,2)/(numpy.power(vprop,2).sum()))*100
        self.inert = iner

        lamb = numpy.power(vprop,2)
        pill = 1/(1 + lamb)
        pillai = numpy.linalg.det(numpy.diag(pill))
        glh = g - 1
        gle = n - g
        t = numpy.sqrt((numpy.power(glh,2) * numpy.power(m,2) - 4)/(numpy.power(m,2) + numpy.power(glh,2) - 5))
        w = gle + glh - 0.5 * (m + glh + 1)
        df1 = m * glh
        df2 = w * t - 0.5 * (m * glh - 2)

        Wilksf = (1 - numpy.power(pillai,1/t))/(numpy.power(pillai,1/t)) * (df2/df1)
        Wilksp = stats.f.pdf(Wilksf, df1, df2)
        self.Wilks = {'f-val': Wilksf,'p-val': Wilksp}

        falfau = stats.t.ppf(1 - (0.025), (n - g))
        falfab = stats.t.ppf(1 - (0.025/(g * m)), (n - g))
        falfam = numpy.sqrt(stats.f.ppf(1 - 0.05, m, (n - g - m + 1)) * (((n - g) * m)/(n - g - m + 1)))
        falfac = 2.447747

        UnivRad = falfau * numpy.diag(numpy.linalg.inv(numpy.sqrt(S11)))/numpy.sqrt(n - g)
        BonfRad = falfab * numpy.diag(numpy.linalg.inv(numpy.sqrt(S11)))/numpy.sqrt(n - g)
        MultRad = falfam * numpy.diag(numpy.linalg.inv(numpy.sqrt(S11)))/numpy.sqrt(n - g)
        ChisRad = falfac * numpy.diag(numpy.linalg.inv(numpy.sqrt(S11)))/numpy.sqrt(n - g)

        self.Radius = {'Uni': UnivRad,'Bonf': BonfRad, 'Mult': MultRad, 'Chis': ChisRad}
    def plot(self, target = None, dim = [0, 1] , xlim=None , ylim= None, figx = 10, figy = 10, arrow_width = 0.1, font_size = 12, radius = 'Bonf', plot_coord = False):
        #Rescale
        sca = numpy.sum(numpy.power(self.Group_Coord[:, :],2)) / len(self.GroupNames)
        scb = numpy.sum(numpy.power(self.Var_Coord[:, dim],2)) / self.data.shape[1]
        scf = numpy.sqrt(numpy.sqrt(scb / sca))
        
        Group_Coord = self.Group_Coord[:, dim] * scf
        Var_Coord = self.Var_Coord[:, dim] / scf
        Ind_Coord = self.Ind_Coord[:, dim] * scf
        Radius = self.Radius[radius]
        
        if xlim is None:
            minx = numpy.min([min(Group_Coord[:,0]), min(Ind_Coord[:,0]), min(Var_Coord[:,0])])
            maxx = numpy.max([max(Group_Coord[:,0]), max(Ind_Coord[:,0]), max(Var_Coord[:,0])])
            xlim = [minx-1, maxx+1]
            
        if ylim is None:
            miny = numpy.min([min(Group_Coord[:,1]), min(Ind_Coord[:,1]), min(Var_Coord[:,1])])
            maxy = numpy.max([max(Group_Coord[:,1]), max(Ind_Coord[:,1]), max(Var_Coord[:,1])])
            ylim = [miny-1, maxy+1]

        circles = []
        for i, el in enumerate(self.GroupNames):
            circles.append(plt.Circle(Group_Coord[i, :], Radius[i], alpha = 0.4))

        fig, ax = plt.subplots(figsize=(figx,figy))

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        if plot_coord == True:
            ax.scatter(Ind_Coord[:, :], Ind_Coord[:, :], c = target)

        for i, circle in enumerate(circles):
            ax.add_artist(circle)
            ax.text(Group_Coord[i, dim[0]],
                    Group_Coord[i, dim[1]],  self.GroupNames[i])

        for i in range(0, Var_Coord.shape[0]):
            ax.arrow(0,0, Var_Coord[i, dim[0]], Var_Coord[i, dim[1]], width= arrow_width)
            ax.text(Var_Coord[i ,dim[0]], 
                    Var_Coord[i,dim[1]] , self.col_names[i], fontsize=font_size)

        plt.show()

class CA(object):
    ''' Benzecri Correspondence Analysis'''
    def __init__(self, data, dim, alpha=1, niter=5, state=0):
        if isinstance(data , pandas.core.frame.DataFrame):
            self.col_names = list(data.columns)
            self.data = data.values
        elif isinstance(data , numpy.ndarray):
            self.col_names = ['Var_'+str(i+1) for i in range(data.shape[1])]
            self.data = data
        else:
            raise ValueError('not pandas DataFrame nor numpy ndarray')

        if isinstance(dim, (int, float)):
            self.dim = dim
        else:
            raise ValueError('not numeric')
        if self.dim>self.data.shape[1]:
            raise ValueError('dim bigger than p')
        
        if (alpha>=0 and alpha<=1):
            self.alpha = alpha
        else:
            raise ValueError('not between 0 and 1')
        
        data = data / data.sum()
        
        dr = numpy.matrix(data.sum(axis=1))
        dc = numpy.matrix(data.sum(axis=0))
        
        data = data - (dr.T).dot(dc)
        
        Dr = numpy.diagflat(1/numpy.sqrt(dr))
        Dc = numpy.diagflat(1/numpy.sqrt(dc))
        data = Dr.dot(data).dot(Dc)
        
        U, Sigma, VT = SVD(data, dim, niter, state)
        
        d = Sigma[:numpy.min(data.shape)]
        r = numpy.min(data.shape)
        
        self.inertia = numpy.power(d,2)*100 / numpy.sum(numpy.power(d,2))
        
        U = Dr.dot(U[:,:r])
        V = Dc.dot(VT.T[:,:r])
        
        D = numpy.diagflat(d)
        A = U.dot(D)
        B = V.dot(D)
        
        sf = numpy.power(A,2).sum(axis = 1)
        cf = numpy.linalg.inv(numpy.diagflat(sf)).dot(numpy.power(A,2))
        
        sc = numpy.power(B,2).sum(axis = 1)
        cc = numpy.linalg.inv(numpy.diagflat(sc)).dot(numpy.power(B,2))
        
        A = U.dot(numpy.diagflat(numpy.power(d,alpha)))
        B = V.dot(numpy.diagflat(numpy.power(d,1-alpha)))
        
        self.AB = A[:, :dim].dot(B[:, :dim].T)
        
        self.eigen_values = numpy.power(d,2)
        
        self.RowCoordinates = A[:, :dim]
        self.ColCoordinates = B[:, :dim]
        
        self.RowContributions = cf[:, :dim] * 100
        self.ColContributions = cc[:, :dim] * 100
        
        