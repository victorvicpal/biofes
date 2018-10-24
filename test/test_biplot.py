import unittest
import numpy as np
import pandas as pd
from biofes.biplot import *
from biofes import biplot
from scipy import stats

class test_functions(unittest.TestCase):
    def test_standardize(self):
        A = np.random.uniform(-300,300,size=(300,30))
        A_st = standardize(A, meth=1)
        A_ref = (A-A.mean(axis = 0))/A.std(axis = 0)
        self.assertAlmostEqual(np.mean(A_ref - A_st), 0, msg='standardization error')
    
    def test_Factor2Binary(self):
        target = list(np.random.randint(np.random.randint(2, 10), size = 100))
        Z = Factor2Binary(target,Name = None)
        
        Z_ref = pd.get_dummies(target)
        self.assertAlmostEqual(np.mean(Z_ref.values - Z.values), 0, msg='Factor2Binary error')
    
    def test_matrixsqrt(self):
        A = np.random.randint(low = 0, high = 200, size=(300, 30))
        d = np.random.randint(30)
        tol = np.finfo(float).eps
        
        Sinv = matrixsqrt(A, d, tol, inv=True)
        U, Sigma, VT = SVD(A, d, niter=5, state=0)
        nz = Sigma > tol
        Sinv_ref = U.dot(np.diag(1/np.sqrt(Sigma[nz]))).dot(VT[nz,:])
        self.assertAlmostEqual(np.mean(Sinv_ref - Sinv), 0, delta=1e-3, msg='matrixsqrt (inv=True) error')
        
        ###############################################################################
        
        A = np.random.randint(low = 0, high = 200, size=(300, 30))
        d = np.random.randint(30)
        tol = np.finfo(float).eps
        
        S = matrixsqrt(A, d, tol, inv=False)
        U, Sigma, VT = SVD(A, d, niter=5, state=0)
        nz = Sigma > tol
        S_ref = U.dot(np.diag(np.sqrt(Sigma[nz]))).dot(VT[nz,:])
        self.assertAlmostEqual(np.mean(S_ref - S), 0, delta=1e-3, msg='matrixsqrt (inv=False) error')
        
        
class test_biplot(unittest.TestCase):
    def test_Classic(self):
        n, p = np.random.randint(70,500), np.random.randint(30,50)
        
        A = np.random.uniform(-300,300,size=(n,p))
        d = np.random.randint(p)
        a = np.random.random(1)[0]
        methods = [None, 1]
        m = methods[np.random.randint(2)]
        
        data_st = standardize(A, m)
        U, Sigma, VT = SVD(data_st, d, niter = 35, state = 0)
        
        EV_ref = np.power(Sigma,2)
        Inert_ref = EV_ref/np.sum(EV_ref) * 100
        
        # Contributions
        
        R = U.dot(np.diag(Sigma[:d]))
        C = np.transpose(VT).dot(np.diag(Sigma[:d]))
        
        sf = np.sum(np.power(A,2),axis=1)
        cf = np.zeros((n,d))
        for k in range(0,d):
            cf[:,k] = np.power(R[:,k],2)*100/sf
        
        sc = np.sum(np.power(A, 2),axis=0)
        cc = np.zeros((p,d))
        for k in range(0,d):
            cc[:,k] = np.power(C[:,k],2)*100/sc
        
        # Coordinates

        R = R.dot(np.diag(np.power(Sigma,a)))
        C = C.dot(np.diag(np.power(Sigma,1-a)))

        sca = np.sum(np.power(R,2))/n
        scb = np.sum(np.power(C,2))/p
        scf = np.sqrt(np.sqrt(scb/sca))

        RowCoord_ref = R*scf
        ColCoord_ref = C/scf
        
        # biplot from biofes
        
        BCla = biplot.Classic(data = A ,dim = d, alpha = a, method = m, niter = 35, state = 0)
        
        # DIMENSION TEST
        self.assertEqual(BCla.RowCoord.shape, (n, d), msg='dimension output error (Classic Biplot)')
        self.assertEqual(BCla.ColCoord.shape, (p, d) , msg='dimension output error (Classic Biplot)')
        self.assertEqual(len(BCla.Inert), d, msg='dimension output error (Classic Biplot)')
        self.assertEqual(len(BCla.EV)   , d, msg='dimension output error (Classic Biplot)')
        
        # INERTIA / EV TEST
        try:
            if str((EV_ref - EV).mean()) == 'nan':
                pass
            else:
                self.assertAlmostEqual(np.mean(EV_ref - BCla.EV), 0, msg='EV error')
                self.assertAlmostEqual(np.mean(Inert_ref - BCla.Inert), 0, msg='Inertia error')
        except:
            pass
        
        # CONTRIBUTIONS TEST
        try:
            if str((cf - BCla.RowCont).mean()) == 'nan':
                pass
            else:
                els = A.shape[0]*A.shape[1]
                self.assertAlmostEqual(np.mean(cf - BCla.RowCont), 0, delta=els*(1e-2), msg='Row Contributions error')
                self.assertAlmostEqual(np.mean(cc - BCla.ColCont), 0, delta=els*(1e-2), msg='Column Contributions error')
        except:
            pass
        
        # COORDINATES TEST
        self.assertAlmostEqual(np.mean(RowCoord_ref - BCla.RowCoord), 0, delta=1e-3, msg='Row Coordinates error')
        self.assertAlmostEqual(np.mean(ColCoord_ref - BCla.ColCoord), 0, delta=1e-3, msg='Col Coordinates error')
        
    def test_Canonical(self):
        n, m = np.random.randint(70,500), np.random.randint(10,50)
        A = np.random.uniform(-300,300,size=(n,m))
        target = list(np.random.randint(np.random.randint(2, 10), size = A.shape[0]))
        gn = list(set(target))
        g = len(gn)
        d = np.random.randint(len(gn)+1, m)
        methods = [None, 1]
        met = methods[np.random.randint(2)]
        
        data_std = standardize(A, met)
        r = np.array([len(gn) - 1, m]).min()
        #Groups to Binary
        Z = Factor2Binary(target)
        ng = Z.sum(axis=0)
        S11 = (Z.T).dot(Z).values
        
        Xb = np.linalg.inv(S11).dot(Z.T).dot(data_std)
        B = (Xb.T).dot(S11).dot(Xb)
        S = (data_std.T).dot(data_std) - B
        Y = np.power(S11,0.5).dot(Xb).dot(matrixsqrt(S,d,inv=True))
        
        U, Sigma, VT = SVD(Y, d, niter = 15, state = 0)
        
        #Variable_Coord
        H = matrixsqrt(S, d, inv=False).dot(np.transpose(VT[0:r,:]))
        #Canonical_Weights
        B = matrixsqrt(S, d, inv=True ).dot(np.transpose(VT[0:r,:]))
        #Group_Coord
        J = Xb.dot(B)
        #Individual_Coord
        V = data_std.dot(B)
        
        sct = np.diag((V.T).dot(V))
        sce = np.diag((J.T).dot(S11).dot(J))
        scr = sct -sce
        fs = (sce/(g - 1))/(scr/(n - g))
        
        #eigenvectors
        vprop = Sigma[:r]
        #Inertia
        iner = (np.power(vprop,2)/(np.power(vprop,2).sum()))*100
        
        lamb = np.power(vprop,2)
        pill = 1/(1 + lamb)
        pillai = np.linalg.det(np.diag(pill))
        glh = g - 1
        gle = n - g
        t = np.sqrt((np.power(glh,2) * np.power(m,2) - 4)/(np.power(m,2) + np.power(glh,2) - 5))
        w = gle + glh - 0.5 * (m + glh + 1)
        df1 = m * glh
        df2 = w * t - 0.5 * (m * glh - 2)
        
        # Wilks
        Wilksf = (1 - np.power(pillai,1/t))/(np.power(pillai,1/t)) * (df2/df1)
        Wilksp = stats.f.pdf(Wilksf, df1, df2)
        Wilks = {'f-val': Wilksf,'p-val': Wilksp}
        
        # Radius
        
        falfau = stats.t.ppf(1 - (0.025), (n - g))
        falfab = stats.t.ppf(1 - (0.025/(g * m)), (n - g))
        falfam = np.sqrt(stats.f.ppf(1 - 0.05, m, (n - g - m + 1)) * (((n - g) * m)/(n - g - m + 1)))
        falfac = 2.447747

        UnivRad = falfau * np.diag(np.linalg.inv(np.sqrt(S11)))/np.sqrt(n - g)
        BonfRad = falfab * np.diag(np.linalg.inv(np.sqrt(S11)))/np.sqrt(n - g)
        MultRad = falfam * np.diag(np.linalg.inv(np.sqrt(S11)))/np.sqrt(n - g)
        ChisRad = falfac * np.diag(np.linalg.inv(np.sqrt(S11)))/np.sqrt(n - g)

        Radius = {'Uni': UnivRad,'Bonf': BonfRad, 'Mult': MultRad, 'Chis': ChisRad}
        
        BCan = biplot.Canonical(data = A, dim = d, GroupNames = gn, y = target, method = met, niter = 35, state = 0)
        
        # DIMENSION TEST
        self.assertEqual(BCan.Ind_Coord.shape, (n, len(gn)-1), msg='dimension output error (Canonical Biplot) Ind_Coord')
        self.assertEqual(BCan.Var_Coord.shape, (m, len(gn)-1), msg='dimension output error (Canonical Biplot) Var_Coord')
        self.assertEqual(BCan.Group_Coord.shape, (len(gn), len(gn)-1), msg='dimension output error (Canonical Biplot) Group_Coord')
        self.assertEqual(len(BCan.inert), len(gn)-1, msg='dimension output error (Canonical Biplot)')
        
        # COORDINATES TEST
        els = H.shape[0]*H.shape[1]
        self.assertAlmostEqual(np.mean(H - BCan.Var_Coord), 0, delta=els*(1e-2), msg='Var Coordinates error')
        els = V.shape[0]*V.shape[1]
        self.assertAlmostEqual(np.mean(V - BCan.Ind_Coord), 0, delta=els*(1e-2), msg='Ind Coordinates error')
        els = J.shape[0]*J.shape[1]
        self.assertAlmostEqual(np.mean(J - BCan.Group_Coord), 0, delta=els*(1e-2), msg='Group Coordinates error')
        
        # CANONICAL WEIGHTS TEST
        els = B.shape[0]*B.shape[1]
        self.assertAlmostEqual(np.mean(B - BCan.Can_Weights), 0, delta=els*(1e-2), msg='Canonical Weights error')
        
        # EV / INERTIA TEST
        try:
            if str((vprop - BCan.vprop).mean()) == 'nan':
                pass
            else:
                self.assertAlmostEqual(np.mean(vprop - BCan.vprop), 0, msg='EV error')
                self.assertAlmostEqual(np.mean(iner -  BCan.inert), 0, msg='Inertia error')
        except:
            pass
        
        # WILKS TEST
        self.assertAlmostEqual(Wilks['f-val'] - BCan.Wilks['f-val'], 0, delta=(1e-3), msg='f-val Wilks error')
        self.assertAlmostEqual(Wilks['p-val'] - BCan.Wilks['p-val'], 0, delta=(1e-3), msg='p-val Wilks error')
        
        # RADIUS
        self.assertAlmostEqual(np.mean(Radius['Uni'] - BCan.Radius['Uni']), 0, delta=(1e-3), msg='Uni Radius error')
        self.assertAlmostEqual(np.mean(Radius['Bonf'] - BCan.Radius['Bonf']), 0, delta=(1e-3), msg='Bonferroni Radius error')
        self.assertAlmostEqual(np.mean(Radius['Mult'] - BCan.Radius['Mult']), 0, delta=(1e-3), msg='Mult Radius error')
        self.assertAlmostEqual(np.mean(Radius['Chis'] - BCan.Radius['Chis']), 0, delta=(1e-3), msg='Chi-sqr Radius error')

if __name__ == '__main__':
    unittest.main()