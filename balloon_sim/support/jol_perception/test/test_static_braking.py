import json

import numpy as np
import pytest

from jol_perception.static_braking import build_braking_landing, sample_braking_landing


def test_measured_pva_brakes_with_C2_join_and_continuous_motion_bounds():
    p,v,a = [10.,3.,6.],[1.1,.15,.08],[.04,-.02,.01]
    plan = build_braking_landing(p,v,a,landing_z=.1,sample_period=100.)
    plan = json.loads(json.dumps(plan,allow_nan=False))
    first = sample_braking_landing(plan,0.)
    for field,expected in [('position',p),('velocity',v),('acceleration',a)]:
        np.testing.assert_allclose(first[field],expected,atol=1e-12)
    join = plan['segments'][1]['start_time_s']
    for field in ('position','velocity','acceleration'):
        np.testing.assert_allclose(sample_braking_landing(plan,join-1e-7)[field],
                                   sample_braking_landing(plan,join+1e-7)[field],atol=1e-6)
    for t in np.linspace(0.,plan['duration_s'],1201):
        row = sample_braking_landing(plan,t)
        for field,key in [('velocity','speed_m_s'),('acceleration','acceleration_m_s2'),('jerk','jerk_m_s3')]:
            assert np.linalg.norm(row[field])<=plan['limits'][key]+1e-9
    t,dt = .43*join,1e-4
    lo,mid,hi = (sample_braking_landing(plan,x) for x in (t-dt,t,t+dt))
    for original,derivative in [('position','velocity'),('velocity','acceleration'),('acceleration','jerk')]:
        np.testing.assert_allclose((np.array(hi[original])-lo[original])/(2*dt),mid[derivative],atol=1e-7)
    end = sample_braking_landing(plan,plan['duration_s'])
    np.testing.assert_allclose(end['position'],plan['goal'])
    assert end['velocity']==end['acceleration']==[0.,0.,0.]
    assert end['phase']=='hold'


def test_braking_rejects_thin_obstacles_target_plane_and_initial_limit_violation():
    args = ([10.,0.,6.],[1.1,0.,0.],[0.,0.,0.])
    plan = build_braking_landing(*args,landing_z=.1,sample_period=100.,clearance=0.)
    middle = np.array(sample_braking_landing(plan,plan['segments'][0]['duration_s']*.53)['position'])
    obstacle = dict(min=(middle-.001).tolist(),max=(middle+.001).tolist())
    with pytest.raises(ValueError,match='obstacle'):
        build_braking_landing(*args,landing_z=.1,sample_period=100.,clearance=0.,obstacles=[obstacle])
    with pytest.raises(ValueError,match='surface_boundary'):
        build_braking_landing(*args,landing_z=.1,surface_boundary=dict(
            target_world=[12.,0.,6.],direction_world=[1.,0.,0.],required_plane_separation_m=.8))
    with pytest.raises(ValueError,match='initial_state'):
        build_braking_landing([0.,0.,3.],[2.,0.,0.],[0.,0.,0.],landing_z=.1)
