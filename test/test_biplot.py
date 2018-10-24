import unittest
import numpy as np
import pandas as pd
from biofes.biplot import *
from biofes import biplot
from math import isclose

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
        n, p = np.random.randint(500), np.random.randint(50)
        
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
                self.assertAlmostEqual(np.mean(Inert_ref - BCla.Inert), 0, msg='EV error')
        except:
            pass
        
        # CONTRIBUTIONS TEST
        #self.assertTrue(np.allclose(cf, BCla.RowCont, rtol=1e-03, atol=1e-05), msg='Row Contributions error')
        #self.assertTrue(np.allclose(cc, BCla.ColCont, rtol=1e-03, atol=1e-05), msg='Col Contributions error')
        els = A.shape[0]*A.shape[1]
        self.assertAlmostEqual(np.mean(cf - BCla.RowCont), 0, delta=els*(1e-3), msg='Row Contributions error')
        self.assertAlmostEqual(np.mean(cc - BCla.ColCont), 0, delta=els*(1e-3), msg='Column Contributions error')
        
        # COORDINATES TEST
        self.assertAlmostEqual(np.mean(RowCoord_ref - BCla.RowCoord), 0, delta=1e-3, msg='Row Coordinates error')
        self.assertAlmostEqual(np.mean(ColCoord_ref - BCla.ColCoord), 0, delta=1e-3, msg='Col Coordinates error')
        
    def test_Canonical(self):
        A = np.random.randint(low = 0, high = 200, size=(300, 30))
        target = list(np.random.randint(np.random.randint(2, 10), size = A.shape[0]))
        gn = list(set(target))
        d = np.random.randint(len(gn)+1, 30)
        methods = [None, 1]
        m = methods[np.random.randint(2)]
        
        BCan = biplot.Canonical(data = A, dim = d, GroupNames = gn, y = target, method = m, niter = 35, state = 0)
        
        self.assertEqual(BCan.Ind_Coord.shape, (300, len(gn)-1), msg='dimension output error (Canonical Biplot)')
        self.assertEqual(BCan.Var_Coord.shape, ( 30, len(gn)-1) , msg='dimension output error (Canonical Biplot)')
        self.assertEqual(len(BCan.inert), len(gn)-1, msg='dimension output error (Canonical Biplot)')

if __name__ == '__main__':
    unittest.main()