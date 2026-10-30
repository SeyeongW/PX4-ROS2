"""Publication paths must not silently resolve to the author's other workspaces."""
from balloon_locator.runtime_paths import runtime_paths


def test_default_paths_follow_the_checked_out_workspace(tmp_path):
    paths = runtime_paths(tmp_path, {})
    root = tmp_path/'external/PX4-Autopilot'
    assert paths == {
        'px4_root': root,
        'px4_binary': root/'build/px4_sitl_default/bin/px4',
        'px4_plugin_dir': root/'build/balloon_gazebo_classic',
    }


def test_explicit_local_build_overrides_are_independent(tmp_path):
    paths = runtime_paths(tmp_path, {
        'BALLOON_PX4_ROOT': str(tmp_path/'source'),
        'BALLOON_PX4_BINARY': str(tmp_path/'custom/px4'),
        'BALLOON_PX4_PLUGIN_DIR': str(tmp_path/'plugins'),
    })
    assert paths['px4_root'] == tmp_path/'source'
    assert paths['px4_binary'] == tmp_path/'custom/px4'
    assert paths['px4_plugin_dir'] == tmp_path/'plugins'
