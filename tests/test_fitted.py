"""Saved-state regression tests with synthetic observations, not forecast-skill tests."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'release'))
import fitted


def panel():
    """Synthetic complete calendar panel containing genuine zeros and lags."""
    random=np.random.default_rng(417)
    rows=[]
    for county in range(24):
        previous=float(random.poisson(4))
        for year in range(2005,2017):
            temperature=random.normal(24,2)
            et=random.uniform(20,75)
            density=random.uniform(.12,.78)
            stand=random.uniform(.1,.9)
            area=random.uniform(20000,150000)
            spots=float(random.poisson(max(.2,2+.08*previous+.02*et)))
            if random.uniform()<.23:
                spots=0.
            rows.append({'FIPS':str(1001+2*county).zfill(5),'YEAR':year,
                         'TMAX_C':temperature,'ET_growing':et,'PINE_DENSITY':density,
                         'STD_1':stand,'PINE_ACRES':area,'PREV_SPOTS':previous,
                         'SPOTS':spots,'PPT_growing':random.uniform(400,1300)})
            previous=spots
    return pd.DataFrame(rows)


class FittedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame=panel()
        cls.model=fitted.fit(cls.frame,2012,{'source':'explicitly synthetic unit-test panel'})

    def test_model_is_json_serializable_and_roundtrip_predictions_identical(self):
        saved=json.loads(json.dumps(self.model,allow_nan=False))
        fitted.validate_model(saved)
        pd.testing.assert_frame_equal(fitted.predict(self.frame,self.model,2016),
                                      fitted.predict(self.frame,saved,2016))

    def test_future_observations_and_predictors_cannot_change_learned_state(self):
        frame=self.frame.copy(deep=True)
        future=frame.YEAR>2012
        frame.loc[future,'SPOTS']=1000000.
        frame.loc[future,['TMAX_C','ET_growing','PINE_DENSITY','STD_1','PINE_ACRES','PREV_SPOTS']]=np.nan
        other=fitted.fit(frame,2012,{'source':'explicitly synthetic unit-test panel'})
        self.assertEqual(self.model,other)
        self.assertLessEqual(other['training']['year_max'],2012)

    def test_zero_observations_are_preserved_and_used(self):
        training=self.frame.loc[self.frame.YEAR<=2012]
        self.assertEqual(self.model['training']['eligible_rows'],len(training))
        self.assertEqual(self.model['training']['zero_spot_rows'],int((training.SPOTS==0).sum()))
        self.assertGreater(self.model['training']['zero_spot_rows'],0)
        result=fitted.predict(self.frame,self.model,2016)
        np.testing.assert_array_equal(result.loc[self.frame.SPOTS==0,'Actual spots'],0.)
        self.assertEqual(self.model['transform']['shift'],1.)
        self.assertIn('YEAR-1',self.model['transform']['lag_definition'])

    def test_scaler_and_risk_reference_use_training_rows_only(self):
        training=self.frame.loc[self.frame.YEAR<=2012]
        for name in fitted.SCALED_COLUMNS:
            self.assertAlmostEqual(self.model['scaler']['mean'][name],training[name].mean())
            self.assertAlmostEqual(self.model['scaler']['std'][name],training[name].std(ddof=1))
        self.assertAlmostEqual(self.model['risk']['mean'],training.SPOTS.mean())
        self.assertAlmostEqual(self.model['risk']['std'],training.SPOTS.std(ddof=1))
        self.assertEqual(self.model['risk']['reference'],'observed eligible training SPOTS')

    def test_scenarios_touch_target_year_only_and_do_not_mutate_inputs_or_state(self):
        original=self.frame.copy(deep=True);state=deepcopy(self.model)
        base=fitted.predict(self.frame,self.model,2016)
        scenario=fitted.predict(self.frame,self.model,2016,tmax=2,et=10,pine=-10)
        pd.testing.assert_frame_equal(self.frame,original)
        self.assertEqual(self.model,state)
        historical=self.frame.YEAR!=2016;target=~historical
        pd.testing.assert_frame_equal(base.loc[historical],scenario.loc[historical])
        np.testing.assert_allclose(scenario.loc[target,'Max April Temperature'],self.frame.loc[target,'TMAX_C']+2)
        np.testing.assert_allclose(scenario.loc[target,'Evap_Transporation'],self.frame.loc[target,'ET_growing']*1.1)
        np.testing.assert_allclose(scenario.loc[target,'Pine_density'],self.frame.loc[target,'PINE_DENSITY']*.9)
        np.testing.assert_array_equal(scenario.Rain,self.frame.PPT_growing)
        np.testing.assert_array_equal(base.Evap_Transporation,self.frame.ET_growing)
        np.testing.assert_array_equal(base.ET_mm_per_month,base.Evap_Transporation)

    def test_prediction_is_independent_of_batch_and_observation_availability(self):
        base=fitted.predict(self.frame,self.model,2016)
        subset=self.frame.loc[self.frame.YEAR==2016].copy()
        subset['SPOTS']=np.nan
        future=fitted.predict(subset,self.model,2016)
        np.testing.assert_allclose(future.SPB_spots,base.loc[subset.index,'SPB_spots'])
        np.testing.assert_allclose(future.Risk_category,base.loc[subset.index,'Risk_category'])
        self.assertTrue(future.Eligible.all())
        self.assertTrue(future['Actual spots'].isna().all())

    def test_missing_predictor_never_becomes_partial_prediction_or_low_risk(self):
        frame=self.frame.iloc[:8].copy()
        frame.loc[frame.index[0],'ET_growing']=np.nan
        frame.loc[frame.index[1],'PREV_SPOTS']=np.nan
        frame.loc[frame.index[2],'PINE_ACRES']=0.
        frame.loc[frame.index[3],'STD_1']=np.inf
        result=fitted.predict(frame,self.model,2005)
        self.assertEqual(len(result),len(frame))
        np.testing.assert_array_equal(result.FIPS,frame.FIPS)
        np.testing.assert_array_equal(result.YEAR,frame.YEAR)
        self.assertTrue(result.iloc[:4].SPB_spots.isna().all())
        self.assertTrue(result.iloc[:4].Risk_category.isna().all())
        self.assertFalse(result.iloc[:4].Eligible.any())
        self.assertTrue(result.iloc[:4].Reason.str.len().gt(0).all())
        self.assertTrue(result.iloc[:4].Prediction_status.eq('ineligible').all())

    def test_invalid_keys_are_retained_with_reasons(self):
        frame=self.frame.iloc[:4].copy()
        frame['FIPS']=frame.FIPS.astype(object)
        frame.loc[frame.index[0],'FIPS']='bad'
        frame['YEAR']=frame.YEAR.astype(object)
        frame.loc[frame.index[1],'YEAR']='bad'
        result=fitted.predict(frame,self.model,2005)
        self.assertEqual(result.iloc[0].FIPS,'bad')
        self.assertEqual(result.iloc[1].YEAR,'bad')
        self.assertIn('invalid_FIPS',result.iloc[0].Reason)
        self.assertIn('invalid_YEAR',result.iloc[1].Reason)
        self.assertTrue(result.iloc[:2].SPB_spots.isna().all())

    def test_duplicate_keys_are_all_excluded_without_adjudication(self):
        frame=pd.concat([self.frame,self.frame.iloc[[0]]],ignore_index=True)
        model=fitted.fit(frame,2012,{'source':'synthetic duplicate fixture'})
        self.assertEqual(model['training']['exclusion_reason_counts']['duplicate_county_year'],2)
        self.assertEqual(model['training']['eligible_rows'],self.model['training']['eligible_rows']-1)
        result=fitted.predict(frame,self.model,2005)
        duplicate=result.duplicated(['FIPS','YEAR'],keep=False)
        self.assertEqual(int(duplicate.sum()),2)
        self.assertTrue(result.loc[duplicate,'SPB_spots'].isna().all())
        self.assertTrue(result.loc[duplicate,'Reason'].str.contains('duplicate_county_year').all())

    def test_unobserved_or_negative_training_labels_are_excluded(self):
        frame=self.frame.copy()
        frame.loc[0,'SPOTS']=np.nan
        frame.loc[1,'SPOTS']=-1
        model=fitted.fit(frame,2012,{'source':'synthetic labels'})
        self.assertEqual(model['training']['eligible_rows'],self.model['training']['eligible_rows']-2)
        self.assertEqual(model['training']['exclusion_reason_counts']['negative_SPOTS'],1)
        self.assertEqual(model['training']['exclusion_reason_counts']['unobserved_or_nonfinite_SPOTS'],1)

    def test_rank_deficiency_and_constant_features_are_rejected(self):
        frame=self.frame.copy()
        frame['TMAX_C']=frame['ET_growing']
        with self.assertRaisesRegex(ValueError,'rank deficient'):
            fitted.fit(frame,2012,{})
        frame=self.frame.copy();frame['PINE_DENSITY']=.5
        with self.assertRaisesRegex(ValueError,'variance'):
            fitted.fit(frame,2012,{})

    def test_no_response_variation_or_too_few_rows_cannot_fit(self):
        frame=self.frame.copy();frame['SPOTS']=0.
        with self.assertRaisesRegex(ValueError,'vary'):
            fitted.fit(frame,2012,{})
        with self.assertRaisesRegex(ValueError,'more than six'):
            fitted.fit(self.frame.iloc[:6],2012,{})

    def test_invalid_artifacts_are_rejected_before_prediction(self):
        edits=[lambda m:m['coefficients'].__setitem__('const',float('nan')),
               lambda m:m['coefficients'].__setitem__('PREV','0.2'),
               lambda m:m['coefficients'].pop('STD_1'),
               lambda m:m['scaler']['std'].__setitem__('ET_growing',0.),
               lambda m:m['transform'].__setitem__('lambda',float('inf')),
               lambda m:m.__setitem__('feature_order',list(reversed(fitted.FEATURE_ORDER))),
               lambda m:m['risk']['thresholds'].__setitem__(1,-1.),
               lambda m:m['risk']['levels'].__setitem__(0,False),
               lambda m:m['training'].__setitem__('design_rank',5),
               lambda m:m['training'].__setitem__('year_max',2013),
               lambda m:m.__setitem__('schema_version',999)]
        for edit in edits:
            with self.subTest(edit=edit):
                model=deepcopy(self.model);edit(model)
                with self.assertRaises(ValueError):fitted.validate_model(model)
                with self.assertRaises(ValueError):fitted.predict(self.frame,model,2016)
        for model in [None,[],{},'model']:
            with self.assertRaises(ValueError):fitted.validate_model(model)

    def test_inverse_domain_failures_stay_missing_and_have_no_risk_category(self):
        model=deepcopy(self.model)
        lmbda=model['transform']['lambda']
        self.assertNotEqual(lmbda,0.)
        model['coefficients']={name:0. for name in model['coefficients']}
        model['coefficients']['const']=-2./lmbda
        fitted.validate_model(model)
        result=fitted.predict(self.frame.iloc[:4],model,2005)
        self.assertTrue(result.SPB_spots.isna().all())
        self.assertTrue(result.Risk_category.isna().all())
        self.assertTrue(result.Reason.str.contains('domain').all())

    def test_negative_inverse_density_is_explicitly_clipped_at_zero(self):
        model=deepcopy(self.model)
        model['coefficients']={name:0. for name in model['coefficients']}
        # For this negative lambda, a small negative linear response is valid
        # but maps to a shifted value below one, hence negative count density.
        model['coefficients']['const']=-.001
        result=fitted.predict(self.frame.iloc[:4],model,2005)
        self.assertTrue(result.Eligible.all())
        np.testing.assert_array_equal(result.SPB_spots,0.)
        self.assertIn('clip negative',model['transform']['inverse_policy'])

    def test_bad_scenarios_are_rejected_or_rows_marked_ineligible(self):
        for keyword in [dict(tmax=np.nan),dict(et=-101),dict(pine=-101)]:
            with self.subTest(keyword=keyword),self.assertRaises(ValueError):
                fitted.predict(self.frame,self.model,2016,**keyword)
        scenario=fitted.predict(self.frame,self.model,2016,pine=10000)
        target=self.frame.YEAR==2016
        self.assertTrue(scenario.loc[target,'SPB_spots'].isna().all())
        self.assertTrue(scenario.loc[target,'Reason'].str.contains('PINE_DENSITY').all())

    def test_absent_scenario_year_is_rejected_instead_of_silent_noop(self):
        with self.assertRaisesRegex(ValueError,'Scenario year 2099 is absent'):
            fitted.predict(self.frame,self.model,2099,tmax=2)

    def test_artifact_provenance_and_schema_are_explicit(self):
        self.assertEqual(self.model['training']['provenance']['source'],'explicitly synthetic unit-test panel')
        self.assertEqual(self.model['training']['design_rank'],6)
        self.assertEqual(len(self.model['training']['data_sha256']),64)
        self.assertEqual(self.model['feature_order'],fitted.FEATURE_ORDER)
        self.assertEqual(set(self.model['coefficients']),{'const',*fitted.FEATURE_ORDER})
        for input_frame,cutoff,provenance in [(self.frame.drop(columns=['PREV_SPOTS']),2012,{}),
                                               (self.frame,2012.5,{}),(self.frame,2012,{'bad':np.nan})]:
            with self.subTest(cutoff=cutoff),self.assertRaises(ValueError):
                fitted.fit(input_frame,cutoff,provenance)


if __name__=='__main__':
    unittest.main()
