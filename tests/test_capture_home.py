"""Protocol home_qpos capture helpers (no robot)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from so101.capture_home import format_home_qpos_line, write_home_qpos_to_protocol
from so101.cli import build_parser
from so101.eval_protocol import load_protocol


def test_format_home_qpos_line() -> None:
    line = format_home_qpos_line([-2.458, -81.9, 74.0, 33.8, -4.7, 5.2])
    assert line.startswith("home_qpos: [")
    assert "-2.458" in line


def test_write_home_qpos_inserts_before_home_steps(tmp_path: Path) -> None:
    p = tmp_path / "proto.yaml"
    p.write_text(
        "\n".join(
            [
                "task: Place keys into metal cup",
                "timeout_s: 60",
                "n_trials_per_condition: 1",
                "home_steps: 80",
                "home_dt: 0.04",
                "success_criterion: done",
                "conditions:",
                "  - id: keys=center",
                "    pen: center",
            ]
        )
        + "\n"
    )
    q = np.array([-1.0, -70.0, 60.0, 30.0, -5.0, 8.0], dtype=np.float32)
    write_home_qpos_to_protocol(p, q)
    proto = load_protocol(p)
    np.testing.assert_allclose(proto.home_qpos, q, atol=1e-3)


def test_write_home_qpos_replaces_existing(tmp_path: Path) -> None:
    p = tmp_path / "proto.yaml"
    p.write_text(
        "task: t\ntimeout_s: 1\nn_trials_per_condition: 1\n"
        "home_qpos: [0, 0, 0, 0, 0, 0]\nsuccess_criterion: x\n"
        "conditions:\n  - id: a\n    pen: a\n"
    )
    write_home_qpos_to_protocol(p, [1, 2, 3, 4, 5, 6])
    proto = load_protocol(p)
    assert proto.home_qpos == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert p.read_text().count("home_qpos:") == 1


def test_capture_home_parser() -> None:
    args = build_parser().parse_args(
        ["capture-home", "--protocol", "config/eval/pen_to_keys.yaml", "--print-only", "-y"]
    )
    assert args.protocol.endswith("pen_to_keys.yaml")
    assert args.print_only is True
    assert args.yes is True
