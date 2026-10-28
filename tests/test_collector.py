"""
Tests — Collector
==================
Verifies that all sub-collectors return dictionaries with the expected
keys and sensible value types. These tests run on the host OS (Windows/Mac/Linux)
since psutil is cross-platform; Linux-only paths are tested via mocking.
"""

import pytest
from heimdall.collector import cpu, memory, disk, network, process, system
from heimdall.collector.snapshot import collect, SystemSnapshot


class TestCpuCollector:
    def test_returns_dict(self):
        result = cpu.collect()
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = cpu.collect()
        for key in ("cpu_percent_mean", "cpu_percent_max", "load_avg_1", "num_logical_cpus"):
            assert key in result, f"Missing key: {key}"

    def test_cpu_pct_range(self):
        result = cpu.collect()
        assert 0.0 <= result["cpu_percent_mean"] <= 100.0

    def test_num_cpus_positive(self):
        result = cpu.collect()
        assert result["num_logical_cpus"] >= 1


class TestMemoryCollector:
    def test_returns_dict(self):
        result = memory.collect()
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = memory.collect()
        for key in ("mem_used_pct", "mem_available_mb", "swap_used_pct"):
            assert key in result

    def test_mem_pct_range(self):
        result = memory.collect()
        assert 0.0 <= result["mem_used_pct"] <= 100.0

    def test_available_mb_nonnegative(self):
        result = memory.collect()
        assert result["mem_available_mb"] >= 0.0


class TestDiskCollector:
    def test_returns_dict(self):
        result = disk.collect()
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = disk.collect()
        for key in ("disk_read_mbps", "disk_write_mbps", "disk_used_pct"):
            assert key in result

    def test_rates_nonnegative(self):
        result = disk.collect()
        assert result["disk_read_mbps"] >= 0.0
        assert result["disk_write_mbps"] >= 0.0


class TestNetworkCollector:
    def test_returns_dict(self):
        result = network.collect()
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = network.collect()
        for key in ("net_bytes_recv_mbps", "net_bytes_sent_mbps", "tcp_connections"):
            assert key in result

    def test_rates_nonnegative(self):
        result = network.collect()
        assert result["net_bytes_recv_mbps"] >= 0.0
        assert result["net_bytes_sent_mbps"] >= 0.0

    def test_connections_nonnegative(self):
        result = network.collect()
        assert result["tcp_connections"] >= 0
        assert result["udp_connections"] >= 0


class TestProcessCollector:
    def test_returns_dict(self):
        result = process.collect()
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = process.collect()
        for key in ("num_processes", "num_threads", "num_zombies", "top_cpu_procs"):
            assert key in result

    def test_counts_positive(self):
        result = process.collect()
        assert result["num_processes"] >= 1
        assert result["num_threads"]   >= 1

    def test_top_procs_structure(self):
        result = process.collect()
        for proc in result["top_cpu_procs"]:
            assert "pid" in proc
            assert "name" in proc
            assert "cpu_pct" in proc


class TestSnapshotCollector:
    def test_collect_returns_snapshot(self):
        snap = collect(label="test")
        assert isinstance(snap, SystemSnapshot)

    def test_snapshot_has_all_subsystems(self):
        snap = collect()
        for attr in ("cpu", "memory", "disk", "network", "process", "system"):
            assert hasattr(snap, attr)
            val = getattr(snap, attr)
            assert isinstance(val, dict)
            assert len(val) > 0

    def test_snapshot_serializable(self):
        import json
        snap = collect()
        json_str = snap.to_json()
        data = json.loads(json_str)
        assert "timestamp" in data
        assert "cpu" in data

    def test_snapshot_roundtrip(self):
        snap = collect(label="test")
        restored = SystemSnapshot.from_dict(snap.to_dict())
        assert restored.timestamp == snap.timestamp
        assert restored.label     == snap.label
