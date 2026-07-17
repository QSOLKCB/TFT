from pathlib import Path

import numpy as np

from tft.cli import main


def test_demo_and_verify_commands(tmp_path: Path, capsys):
    output = tmp_path / "demo"
    assert (
        main(
            [
                "demo",
                "--out",
                str(output),
                "--sample-rate",
                "8000",
                "--duration",
                "0.02",
                "--fmax",
                "1000",
            ]
        )
        == 0
    )
    assert (output / "manifest.json").is_file()
    assert main(["verify", str(output)]) == 0
    assert "PASS" in capsys.readouterr().out


def test_custom_csv_command(tmp_path: Path):
    tensor_path = tmp_path / "tensor.csv"
    np.savetxt(tensor_path, np.diag([1.0, 2.0, 3.0]), delimiter=",")
    output = tmp_path / "custom"

    exit_code = main(
        [
            "run",
            "--tensor",
            str(tensor_path),
            "--out",
            str(output),
            "--sample-rate",
            "8000",
            "--duration",
            "0.02",
            "--fmax",
            "1000",
        ]
    )
    assert exit_code == 0
    assert (output / "tensor.npy").is_file()


def test_cli_reports_invalid_matrix(tmp_path: Path, capsys):
    tensor_path = tmp_path / "bad.csv"
    np.savetxt(tensor_path, np.ones((2, 3)), delimiter=",")
    exit_code = main(["run", "--tensor", str(tensor_path), "--out", str(tmp_path / "bad")])
    assert exit_code == 2
    assert "must be square" in capsys.readouterr().err
