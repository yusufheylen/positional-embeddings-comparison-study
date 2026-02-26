import os
import shutil
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = REPO_ROOT / "scripts" / "run_evals.sh"


def _make_temp_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    shutil.copy2(SOURCE_SCRIPT, project / "scripts" / "run_evals.sh")
    script_path = project / "scripts" / "run_evals.sh"
    script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR)
    return project


def _make_python_stub(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "python"
    stub.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$@\" > \"$CAPTURE_FILE\"\n"
    )
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    return bin_dir


def _run_script(project: Path, args: list[str], tmp_path: Path):
    capture_file = tmp_path / "captured-argv.txt"
    env = os.environ.copy()
    env["PATH"] = f"{_make_python_stub(tmp_path)}:{env.get('PATH', '')}"
    env["CAPTURE_FILE"] = str(capture_file)

    proc = subprocess.run(
        ["bash", str(project / "scripts" / "run_evals.sh"), *args],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
    )
    captured = capture_file.read_text().splitlines() if capture_file.exists() else []
    return proc, captured


def test_context_lengths_accepts_multiple_unquoted_values_and_relative_checkpoint(tmp_path):
    project = _make_temp_project(tmp_path)
    checkpoint = project / "outputs" / "scaffold_rope_10k"
    checkpoint.mkdir(parents=True)

    proc, argv = _run_script(
        project,
        ["--context-lengths", "32", "64", "./outputs/scaffold_rope_10k"],
        tmp_path,
    )

    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "--checkpoint" in argv
    assert argv[argv.index("--checkpoint") + 1] == "./outputs/scaffold_rope_10k"
    assert argv[argv.index("--pe-type") + 1] == "nope"
    idx = argv.index("--context-lengths")
    assert argv[idx + 1 : idx + 3] == ["32", "64"]


def test_checkpoint_path_with_spaces_and_semicolon_is_passed_as_single_arg(tmp_path):
    project = _make_temp_project(tmp_path)
    weird_parent = tmp_path / "dir with spaces;semi"
    checkpoint = weird_parent / "run1_rope"
    checkpoint.mkdir(parents=True)

    proc, argv = _run_script(project, [str(checkpoint)], tmp_path)

    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert argv[argv.index("--checkpoint") + 1] == str(checkpoint)
    assert argv[argv.index("--pe-type") + 1] == "rope"


def test_missing_eval_value_returns_clear_error(tmp_path):
    project = _make_temp_project(tmp_path)

    proc, argv = _run_script(project, ["--eval"], tmp_path)

    assert proc.returncode != 0
    assert argv == []
    combined = proc.stderr + proc.stdout
    assert "--eval requires a value" in combined
