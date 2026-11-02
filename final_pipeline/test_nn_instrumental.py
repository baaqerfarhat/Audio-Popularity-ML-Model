import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from nn_instrumental import Features, Network, splits, label_features, losses


class NeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df=pd.read_csv(Path(__file__).parent.parent/'data/train.csv').dropna(subset=['uri']).iloc[:150].reset_index(drop=True)
        cls.y=cls.df.Popularity_Type.eq('High').astype(int).to_numpy()

    def test_group_separation(self):
        for a,b in splits(self.df,self.y,7):
            self.assertFalse(set(self.df.uri.iloc[a]) & set(self.df.uri.iloc[b]))

    def test_query_isolation_and_alignment(self):
        a,b=splits(self.df,self.y,7)[0]
        feature=Features(); x=feature.fit_transform(self.df.iloc[a],self.y[a],8)
        q=self.df.iloc[b].copy(); p=feature.transform(q)
        q['Popularity_Type']='unavailable'
        np.testing.assert_array_equal(p,feature.transform(q))
        np.testing.assert_allclose(p[::-1],feature.transform(q.iloc[::-1]),rtol=1e-6,atol=1e-6)
        old=feature.scaler.mean_.copy(); q.iloc[0,q.columns.get_loc('tempo')]=1e10
        changed=feature.transform(q)
        np.testing.assert_array_equal(old,feature.scaler.mean_)
        np.testing.assert_array_equal(p[1:],changed[1:])
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue(np.isfinite(changed).all())

    def test_reference_labels_only(self):
        a,b=splits(self.df,self.y,7)[0]
        q=self.df.iloc[b].copy()
        p=label_features(self.df.iloc[a],self.y[a],q)
        q['Popularity_Type']='changed'
        np.testing.assert_array_equal(p,label_features(self.df.iloc[a],self.y[a],q))

    def test_zero_initialization_routing_and_gradient(self):
        torch.manual_seed(1); model=Network(4,True); model.eval()
        x=torch.randn(6,4); mask=torch.tensor([0.,1.,0.,1.,0.,1.]); y=torch.tensor([0.,1.,1.,0.,0.,1.])
        routed,g,r=model(x,mask)
        torch.testing.assert_close(r,torch.zeros_like(r)); torch.testing.assert_close(routed,g)
        with torch.no_grad():
            model.residual[-1].bias.fill_(2.)
        routed,g,r=model(x,mask)
        torch.testing.assert_close(routed[mask==0],g[mask==0])
        torch.testing.assert_close(routed[mask==1],g[mask==1]+2)
        losses(model,x,y,mask)[0].backward()
        self.assertIsNotNone(model.general.weight.grad)
        self.assertGreater(float(model.residual[-1].bias.grad.abs().sum()),0)


if __name__=='__main__':
    unittest.main()
