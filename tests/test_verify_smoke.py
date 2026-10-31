"""Exercise result decoding, validation, and plotting without running training."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from postprocessing.recorder import PythonObjectEncoder, Recorder
from verify_smoke import VerificationError, check_results


class SmokeResultTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.results = self.root / "results"
        self.results.mkdir()
        self.output = self.root / "smoke.png"
        self.addCleanup(plt.close, "all")

    def result(self):
        recorder = Recorder(["client-0", "client-1"])
        recorder.res["hyperparam"].update({
            "dataset_id": "MNIST", "model_name": "LeNet", "fed_method": "ResFed",
            "num_client": 2, "num_comm_round": 3, "num_epoch": 1,
            "num_local_class": 10, "bs": 64, "lr": 0.01, "i_seed": 1,
            "ul_spar": 0.95, "dl_spar": 0.95, "resfed_upstream": 1,
            "resfed_downstream": 1, "ul_proto_id": 1, "dl_proto_id": 1,
            "qe": 1, "Param_01": 0,
        })
        # Training stores NumPy scalar arrays using the custom pickle-backed encoder.
        recorder.res["server"]["iid_accuracy"] = [np.array(value, dtype=np.float32) for value in (0.2, 0.4, 0.6)]
        recorder.res["server"]["train_loss"] = [np.array(value, dtype=np.float32) for value in (2.0, 1.5, 1.0)]
        for client in recorder.res["clients"].values():
            client["ul_bits"] = [1, 25.0, 26.0]
            client["dl_bits"] = [1, 0, 27.0]
        return recorder.res

    def save(self, result):
        path = self.results / "smoke_result"
        with path.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, cls=PythonObjectEncoder)
        return path

    def test_custom_encoded_result_and_warmup_generate_three_panel_png(self):
        result_file = self.save(self.result())
        self.assertIn("_python_object", result_file.read_text(encoding="utf-8"))
        plot = Recorder.plot
        plotted_axes = []

        def capture_plot(recorder):
            plot(recorder)
            plotted_axes.extend(plt.gcf().axes)

        with patch.object(Recorder, "plot", new=capture_plot), contextlib.redirect_stdout(io.StringIO()):
            check_results(self.results, self.output, np)

        self.assertEqual(len(plotted_axes), 3)
        self.assertEqual(self.output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(plt.get_fignums(), [])

    def test_nonfinite_metric_and_incomplete_series_do_not_write_plot(self):
        cases = (
            ("iid_accuracy", [np.float32(0.2), np.float32("nan"), np.float32(0.6)]),
            ("train_loss", [np.float32(2.0), np.float32(1.5)]),
        )
        for key, values in cases:
            with self.subTest(metric=key):
                result = self.result()
                result["server"][key] = values
                self.save(result)
                with self.assertRaisesRegex(VerificationError, "three finite values"):
                    check_results(self.results, self.output, np)
                self.assertFalse(self.output.exists())

    def test_plot_inside_result_directory_is_rejected_before_writing(self):
        self.save(self.result())
        output = self.results / "smoke.png"
        with self.assertRaisesRegex(VerificationError, "outside the result-input directory"):
            check_results(self.results, output, np)
        self.assertFalse(output.exists())

    def test_missing_and_empty_result_directories_explain_next_step(self):
        cases = ((self.root / "missing", "does not exist"), (self.results, "No result files"))
        for directory, message in cases:
            with self.subTest(directory=directory):
                with self.assertRaisesRegex(VerificationError, message + ".*Run the smoke experiment first"):
                    check_results(directory, self.output, np)
                self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
