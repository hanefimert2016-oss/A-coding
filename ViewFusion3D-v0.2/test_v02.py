from v02 import suggest, Config, run
def test_scheduler():
    assert sorted(suggest([0,90,180,270],45)) == [45.0,135.0,225.0,315.0]
def test_cpu_demo(tmp_path):
    r=run(tmp_path,Config(size=72,stride=4,target_max_gap_deg=45,max_rounds=1,voxel_size=.04))
    assert r['fused_points']>300
    assert len(r['final_angles'])==8
    assert (tmp_path/'helmet_mesh.glb').exists()
