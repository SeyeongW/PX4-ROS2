"""Experiment reuse must preserve recorded data, including failed copies."""
import json
from unittest.mock import patch

import pytest

from jol_perception.static_flight import archive_records


def test_archive_preserves_flights_and_unrelated_user_files(tmp_path):
    previous = {'telemetry.jsonl': '{"stamp":1}\n', 'report.json': json.dumps({'arrived': True}),
                'landing_plan.json': '{"start_stamp":2}', 'plan.json': '{"valid":true}'}
    for name, value in previous.items(): (tmp_path/name).write_text(value)
    (tmp_path/'user_notes.txt').write_text('keep')
    archive = archive_records(tmp_path)
    assert {path.name: path.read_text() for path in archive.iterdir()} == previous
    assert not any((tmp_path/name).exists() for name in previous)
    assert (tmp_path/'user_notes.txt').read_text() == 'keep'
    (tmp_path/'report.json').write_text('{"arrived":false}')
    second = archive_records(tmp_path)
    assert second != archive and json.loads((archive/'report.json').read_text())['arrived']
    assert not json.loads((second/'report.json').read_text())['arrived']
    assert archive_records(tmp_path) is None


def test_copy_failure_keeps_all_original_records(tmp_path):
    (tmp_path/'plan.json').write_text('plan')
    (tmp_path/'report.json').write_text('report')
    with patch('jol_perception.static_flight.shutil.copy2', side_effect=OSError('disk error')):
        with pytest.raises(OSError, match='disk error'): archive_records(tmp_path)
    assert (tmp_path/'plan.json').read_text() == 'plan'
    assert (tmp_path/'report.json').read_text() == 'report'
