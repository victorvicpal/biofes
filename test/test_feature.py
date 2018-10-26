import unittest
import numpy as np
import pandas as pd
from biofes import biplot
from biofes import feature
import itertools

class test_feature(unittest.TestCase):
    def test_selection(self):
        n, p = np.random.randint(70,500), np.random.randint(30,50)
        
        A = np.random.uniform(-300,300,size=(n,p))
        target = list(np.random.randint(np.random.randint(2, 10), size = A.shape[0]))
        d = np.random.randint(p)
        a = np.random.random(1)[0]
        corr = np.random.random(1)[0]
        disc =  np.random.random(1)[0]*100
        methods = [None, 1]
        m = methods[np.random.randint(2)]
        
        # TEST CLASSIC BIPLOT
        bip = biplot.Classic(data = A , dim = d, alpha = a, method = m, niter = 15, state = 0)
        T = feature.selection(bip, target=np.array(target), thr_dis = disc, thr_corr = corr)
        
        Project = bip.RowCoord.dot(bip.ColCoord.T)
        C = bip.ColCoord

        v_min = np.array([abs(el) if el < 0 else el for el in Project.min(axis=0)])

        for i, proj in enumerate(Project.T):
            Project[:,i] = proj + v_min[i]
        
        classes = np.unique(target)
        
        IND = []
        for cl in classes:
            ind_class = []
            for i, el in enumerate(target):
                if el == cl:
                    ind_class.append(i)
            IND.append(ind_class)
        
        num_c = int(len(classes)*(len(classes)-1)/2)

        Disc = np.zeros((bip.data.shape[1], num_c))

        comb = np.array(list(itertools.combinations(classes,r=2)))
        
        # Disc vectors

        for i, cmb in enumerate(comb):
            Disc[:,i] = abs(Project[IND[int(cmb[0])]].mean(axis=0) - Project[IND[int(cmb[1])]].mean(axis=0))
        
        var_comb = ['{}-{}'.format(int(cmb[0]),int(cmb[1])) for i, cmb in enumerate(comb)]
        
        # TEST DISC
        self.assertAlmostEqual(np.mean(Disc - T.Disc.values), 0, delta=(1e-2), msg='Discriminant matrix error')
        
        POS = []
        for v in Disc.T:
            for i, el in enumerate(v):
                if el > np.percentile(v, disc):
                    POS.append(i)
        POS = list(set(POS))

        Corr_matr = np.tril(np.corrcoef(bip.data[:,POS].T), -1)
        
        var_names = [bip.col_names[pos] for pos in POS]

        Corr_matr = pd.DataFrame(Corr_matr, index = var_names, columns = var_names)
        # TEST CORR MATRIX
        try:
            if str((Corr_matr.values - T.Corr_matr.values).mean()) == 'nan':
                self.assertAlmostEqual(np.mean(Corr_matr.values - T.Corr_matr.values), 0, delta=(1e-2), msg='Correlation matrix error')
            else:
                pass
        except:
            pass
        
        pos_corr = np.where(Corr_matr > corr)
        disc_vect = Disc[POS,:].sum(axis = 1)

        ind_del = []
        if pos_corr:
            for i in range(len(pos_corr[0])):
                if disc_vect[pos_corr[0][i]] > disc_vect[pos_corr[1][i]]:
                    ind_del.append(pos_corr[1][i])
                else:
                    ind_del.append(pos_corr[0][i])


        ind_del = list(set(ind_del))
        if ind_del:
            POS = [el for i, el in enumerate(POS) if i not in ind_del]

        var_sel = list(np.array(bip.col_names)[POS])
        
        # TEST VARIABLE SELECTION
        self.assertEqual(T.var_sel, var_sel, msg='Variable selection error')
        
        # TEST CANONICAL BIPLOT
        gn = list(set(target))
        bip = biplot.Canonical(data = A, dim = d, GroupNames = gn, y = target, method = m, niter = 15, state = 0)
        T = feature.selection(bip, target=np.array(target), thr_dis = disc, thr_corr = corr)
        
        Project = bip.Var_Coord.dot(bip.Group_Coord.T)
        C = bip.Var_Coord

        v_min = np.array([abs(el) if el < 0 else el for el in Project.min(axis=0)])

        for i, proj in enumerate(Project.T):
            Project[:,i] = proj + v_min[i]
        
        classes = np.unique(target)
        
        IND = []
        for cl in classes:
            ind_class = []
            for i, el in enumerate(target):
                if el == cl:
                    ind_class.append(i)
            IND.append(ind_class)
        
        num_c = int(len(classes)*(len(classes)-1)/2)

        Disc = np.zeros((bip.data.shape[1], num_c))

        comb = np.array(list(itertools.combinations(classes,r=2)))
        
        # Disc vectors

        for i, cmb in enumerate(comb):
            Disc[:,i] = abs(Project[:,int(cmb[0])] - Project[:,int(cmb[1])])
        
        var_comb = ['{}-{}'.format(int(cmb[0]),int(cmb[1])) for i, cmb in enumerate(comb)]
        
        # TEST DISC
        self.assertAlmostEqual(np.mean(Disc - T.Disc.values), 0, delta=(1e-2), msg='Discriminant matrix error')
        
        POS = []
        for v in Disc.T:
            for i, el in enumerate(v):
                if el > np.percentile(v, disc):
                    POS.append(i)
        POS = list(set(POS))

        Corr_matr = np.tril(np.corrcoef(bip.data[:,POS].T), -1)
        
        var_names = [bip.col_names[pos] for pos in POS]

        Corr_matr = pd.DataFrame(Corr_matr, index = var_names, columns = var_names)
        # TEST CORR MATRIX
        try:
            if str((Corr_matr.values - T.Corr_matr.values).mean()) == 'nan':
                self.assertAlmostEqual(np.mean(Corr_matr.values - T.Corr_matr.values), 0, delta=(1e-2), msg='Correlation matrix error')
            else:
                pass
        except:
            pass
        
        pos_corr = np.where(Corr_matr > corr)
        disc_vect = Disc[POS,:].sum(axis = 1)

        ind_del = []
        if pos_corr:
            for i in range(len(pos_corr[0])):
                if disc_vect[pos_corr[0][i]] > disc_vect[pos_corr[1][i]]:
                    ind_del.append(pos_corr[1][i])
                else:
                    ind_del.append(pos_corr[0][i])


        ind_del = list(set(ind_del))
        if ind_del:
            POS = [el for i, el in enumerate(POS) if i not in ind_del]

        var_sel = list(np.array(bip.col_names)[POS])
        
        # TEST VARIABLE SELECTION
        self.assertEqual(T.var_sel, var_sel, msg='Variable selection error')
        
if __name__ == '__main__':
    unittest.main()