import unittest
import numpy as np
from biofes import biplot

class test_biplot(unittest.TestCase):
    def test_Classic_dim(self):
        A = np.random.randint(low = 0, high = 200, size=(300, 30))
        d = np.random.randint(30)
        a = np.random.random(1)[0]
        methods = [None, 1]
        m = methods[np.random.randint(2)]
        
        BCla = biplot.Classic(data = A ,dim = d, alpha = a, method = m, niter = 5, state = 0)
        
        self.assertEqual(BCla.RowCoord.shape, (300, d), msg='dimension output error (Classic Biplot)')
        self.assertEqual(BCla.ColCoord.shape, ( 30, d) , msg='dimension output error (Classic Biplot)')
        self.assertEqual(len(BCla.Inert), d, msg='dimension output error (Classic Biplot)')
        self.assertEqual(len(BCla.EV)   , d, msg='dimension output error (Classic Biplot)')
        
    def test_Canonical_dim(self):
        A = np.random.randint(low = 0, high = 200, size=(300, 30))
        d = np.random.randint(30)
        target = list(np.random.randint(3, size = A.shape[0]))
        gn = list(set(target))
        methods = [None, 1]
        m = methods[np.random.randint(2)]
        
        BCan = biplot.Canonical(data = A, dim = d, GroupNames = gn, y = target, method = m, niter = 5, state = 0)
        
        self.assertEqual(BCan.Ind_Coord.shape, (300, len(gn)-1), msg='dimension output error (Canonical Biplot)')
        self.assertEqual(BCan.Var_Coord.shape, ( 30, len(gn)-1) , msg='dimension output error (Canonical Biplot)')
        self.assertEqual(len(BCan.inert), len(gn)-1, msg='dimension output error (Canonical Biplot)')

if __name__ == '__main__':
    unittest.main()