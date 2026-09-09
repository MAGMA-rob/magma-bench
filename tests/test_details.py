"""Compiled artifact version compatibility at the JSON boundary."""
import json

import pytest
from pydantic import ValidationError

from magma_bench.artifacts.models import BenchmarkManifest, load_json_model


@pytest.mark.parametrize('version', ['1.5', '1.6'])
def test_supported_manifest_versions_load(tmp_path, version):
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'schema_version': version, 'benchmark_version': 'test',
        'generated_at': '2026-01-01', 'semantic_variations': 1, 'scenarios': [], 'episodes': []}))
    manifest = load_json_model(path, BenchmarkManifest)
    assert manifest.schema_version == version


def test_unknown_manifest_version_is_rejected():
    with pytest.raises(ValidationError, match='schema_version'):
        BenchmarkManifest(schema_version='unsupported', benchmark_version='test',
            generated_at='2026-01-01', semantic_variations=1, scenarios=[], episodes=[])
