import unittest
import numpy as np
import pandas as pd
from try_era_features import EraFeatures,COLUMNS,decades


def fixture():
    frame=pd.DataFrame({'uri':['u'+str(i) for i in range(80)],'track_album_release_date':['1999']*40+['2020-01-01']*40})
    for c in COLUMNS:
        frame[c]=np.arange(80,dtype=float)
    return frame


class EraTests(unittest.TestCase):
    def test_decade_boundaries(self):
        frame=pd.DataFrame({'track_album_release_date':['1999','2000-01','2020-01-01','bad',None]})
        np.testing.assert_equal(decades(frame),[1990,2000,2020,np.nan,np.nan])

    def test_self_exclusion_dedup_and_midrank(self):
        df=fixture(); fitted=EraFeatures().fit(pd.concat([df,df.iloc[[0]]]))
        p,f,n=fitted.transform(df.iloc[[20]])
        np.testing.assert_allclose(p,20/39); self.assertFalse(f[0]); self.assertEqual(n[0],39)
        query=df.iloc[[20]].copy(); query['uri']='new'
        np.testing.assert_allclose(fitted.transform(query)[0],20.5/40)

    def test_fallback_missing_and_query_isolation(self):
        df=fixture(); fitted=EraFeatures().fit(df)
        query=df.iloc[[20,30]].copy(); query['uri']=['new1','new2']; query['track_album_release_date']='1900'
        p,f,n=fitted.transform(query); self.assertTrue(f.all()); np.testing.assert_allclose(p.iloc[0],20.5/80)
        query['Popularity_Type']=['High','Low']; pd.testing.assert_frame_equal(p,fitted.transform(query)[0])
        query.loc[query.index[0],COLUMNS]=1e9
        pd.testing.assert_series_equal(p.iloc[1],fitted.transform(query)[0].iloc[1])
        query.loc[query.index[0],'loudness']=np.nan
        self.assertTrue(np.isnan(fitted.transform(query)[0].iloc[0,0]))
        pd.testing.assert_frame_equal(fitted.reference[COLUMNS],df[COLUMNS])


if __name__=='__main__':
    unittest.main()
