import unittest
import numpy as np
from sklearn.metrics import roc_auc_score
from try_period_specialists import release_periods, route_predictions, pairwise_auc


class PeriodTests(unittest.TestCase):
    def test_boundaries_and_unknown_fallback(self):
        periods=release_periods(["2019-12-31","2020-01-01","2024","1998-03",None,"bad"])
        np.testing.assert_array_equal(periods,[0,1,1,0,-1,-1])
        result=route_predictions(np.full(6,.4),np.full(6,.2),np.full(6,.8),periods)
        np.testing.assert_array_equal(result,[.2,.8,.8,.2,.4,.4])

    def test_pair_blocks_reconcile_weighted_auc_with_ties(self):
        y=np.array([0,1,0,1,0,1,0,1]); periods=np.array([0,0,0,0,1,1,1,1])
        p=np.array([.1,.2,.5,.5,.8,.3,.6,.9]); w=np.arange(1,9,dtype=float)
        numerator=denominator=0.
        for a in [0,1]:
            for b in [0,1]:
                auc,mass=pairwise_auc(y,p,w,a,b,periods)
                selected=((y==1)&(periods==a))|((y==0)&(periods==b))
                self.assertAlmostEqual(auc,roc_auc_score(y[selected],p[selected],sample_weight=w[selected]))
                numerator+=auc*mass; denominator+=mass
        self.assertAlmostEqual(numerator/denominator,roc_auc_score(y,p,sample_weight=w))


if __name__ == "__main__":
    unittest.main()
