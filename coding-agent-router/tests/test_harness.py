import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from benchmark.harness import extract_patch, parse_eval_result
from benchmark.sample_issues import stratified_sample


# ---------------------------------------------------------------------------
# harness.extract_patch
# ---------------------------------------------------------------------------

def test_extract_patch_returns_diff(tmp_path):
    # Init a git repo, add a file, stage it but leave it unstaged-modified
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    f = tmp_path / "foo.py"
    f.write_text("x = 1\n")
    subprocess.run(["git", "add", "foo.py"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True
    )
    f.write_text("x = 2\n")

    patch = extract_patch(tmp_path)
    assert "foo.py" in patch
    assert "-x = 1" in patch
    assert "+x = 2" in patch


def test_extract_patch_empty_on_clean_repo(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t.com"], cwd=tmp_path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "T"], cwd=tmp_path, check=True, capture_output=True
    )
    f = tmp_path / "bar.py"
    f.write_text("pass\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True
    )

    assert extract_patch(tmp_path) == ""


# ---------------------------------------------------------------------------
# harness.parse_eval_result
# ---------------------------------------------------------------------------

def test_parse_eval_result_resolved(tmp_path):
    results_dir = tmp_path / "some-run" / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "results.json").write_text(
        json.dumps({"resolved": ["sympy__sympy-12345"]})
    )
    result = parse_eval_result("sympy__sympy-12345", "some-run", tmp_path)
    assert result["passed"] is True


def test_parse_eval_result_not_resolved(tmp_path):
    results_dir = tmp_path / "run2" / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "results.json").write_text(json.dumps({"resolved": []}))
    result = parse_eval_result("django__django-99999", "run2", tmp_path)
    assert result["passed"] is False


def test_parse_eval_result_missing_file(tmp_path):
    result = parse_eval_result("any-id", "run3", tmp_path)
    assert result["passed"] is False
    assert result["raw"] == {}


# ---------------------------------------------------------------------------
# sample_issues.stratified_sample
# ---------------------------------------------------------------------------

def _make_instances(repos, n_each=5):
    instances = []
    for repo in repos:
        for i in range(n_each):
            instances.append({
                "instance_id": f"{repo.replace('/', '__')}-{i}",
                "repo": repo,
                "base_commit": "abc123",
                "problem_statement": "x" * (200 + i * 300),
                "FAIL_TO_PASS": json.dumps(["test_foo"] * (1 + i % 4)),
            })
    return instances


def test_stratified_sample_count():
    instances = _make_instances(
        ["sympy/sympy", "django/django", "scikit-learn/scikit-learn",
         "pallets/flask", "psf/requests", "pytest-dev/pytest"],
        n_each=10,
    )
    sample = stratified_sample(instances, 50, seed=42)
    assert len(sample) == 50


def test_stratified_sample_min_repos():
    instances = _make_instances(
        ["sympy/sympy", "django/django", "scikit-learn/scikit-learn",
         "pallets/flask", "psf/requests"],
        n_each=15,
    )
    sample = stratified_sample(instances, 50, seed=0)
    repos = {i["repo"] for i in sample}
    assert len(repos) >= 5


def test_stratified_sample_reproducible():
    instances = _make_instances(
        ["sympy/sympy", "django/django", "scikit-learn/scikit-learn",
         "pallets/flask", "psf/requests"],
        n_each=15,
    )
    s1 = [i["instance_id"] for i in stratified_sample(instances, 50, seed=7)]
    s2 = [i["instance_id"] for i in stratified_sample(instances, 50, seed=7)]
    assert s1 == s2


# ---------------------------------------------------------------------------
# WorkQueue
# ---------------------------------------------------------------------------

def test_work_queue_claim_and_complete(tmp_path):
    from benchmark.run_swe_bench import WorkQueue

    q = WorkQueue(tmp_path / "queue")
    q.populate(["a", "b", "c"])

    claimed = set()
    while True:
        iid = q.claim()
        if iid is None:
            break
        claimed.add(iid)
        q.complete(iid)

    assert claimed == {"a", "b", "c"}
    assert q.done_count() == 3
    assert q.remaining() == 0


def test_work_queue_requeue(tmp_path):
    from benchmark.run_swe_bench import WorkQueue

    q = WorkQueue(tmp_path / "queue")
    q.populate(["x"])
    iid = q.claim()
    assert iid == "x"
    q.requeue(iid)
    assert q.remaining() == 1
    assert q.in_progress() == 0


# ---------------------------------------------------------------------------
# Trajectory persistence (server integration)
# ---------------------------------------------------------------------------

def test_trajectory_flushed_to_disk(tmp_path, monkeypatch):
    import proxy.server as srv

    monkeypatch.setattr(srv, "_traj_dir", tmp_path)
    traj = srv.trajectory_store.get_or_create("test-flush-session")
    traj.record_step(
        request={"messages": []},
        response={"choices": [{}], "usage": {}},
        backend="local",
        decision_reason="static",
        latency_s=0.1,
    )
    srv._flush_trajectory(traj)

    written = list(tmp_path.glob("*.json"))
    assert len(written) == 1
    data = json.loads(written[0].read_text())
    assert data["id"] == "test-flush-session"
    assert len(data["steps"]) == 1
