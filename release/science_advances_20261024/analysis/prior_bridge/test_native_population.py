"""Synthetic parser/provenance tests; no simulator or human-data inference."""
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_population as n


def configuration(amplitude=0, gain=1, seed=123):
    # 41 synthetic configured edges, including the two external input columns.
    edges = [(j, i) for i in range(1, 10) for j in range(1, 12)][:41]
    text = "/* SYNTHETIC SOFTWARE FIXTURE */\nTime: 20 Deltat: .0001\nNodes: 1\nConnection matrix:\nFrom: " + " ".join(map(str, range(1, 12))) + "\n"
    for destination in range(1, 12):
        text += f"To {destination}: " + " ".join(str(edges.index((source, destination)) + 1) if (source, destination) in edges else "0" for source in range(1, 12)) + "\n"
    for population in range(1, 10):
        text += f"Population {population}: Synthetic{population}\nQ: 1\n"
        for number, (source, destination) in enumerate(edges, 1):
            if destination == population:
                text += f"Dendrite {number}: Scaled - alpha: 2 beta: 3 hsyn: {gain}\n"
    text += f"Population 10: Stimulation\nStimulus: White - Onset: 0 Mean: 2 StdDev: 1 WhiteSeed: {seed}\n"
    text += f"Population 11: Perturb\nStimulus: PulseRect - Onset: 12 Amplitude: {amplitude} Width: .02 Pulses: 1\n"
    for number in range(1, 42):
        text += f"Propagator {number}: Map - Tau: 0\n"
    for number in range(1, 42):
        text += f"Coupling {number}: Map - nu:\n.001\n"
    text += "Output: Node: All Start: 10 Interval: .002\nPopulation: 1 2 3 4 5 6 7 8 9 10 11\n"
    return text


def native_raw(amplitude=0, mutate=None, columns=None, header_node="1"):
    columns = ["Time", *n.COLUMNS, "Pop.10.Q", "Pop.11.Q"] if columns is None else columns
    times = np.arange(1, n.N_ROWS + 1) * n.DT
    arrays = {"Time": times, "Pop.10.Q": np.cos(times * 17), "Pop.11.Q": amplitude * ((times >= 2) & (times < 2.02))}
    for i, name in enumerate(n.COLUMNS):
        arrays[name] = np.sin(times * (i + 1)) + i / 10
    data = np.column_stack([arrays[name] for name in columns])
    if mutate:
        mutate(data, columns)
    import io
    stream = io.StringIO()
    np.savetxt(stream, data, fmt="%.12g")
    return ("SYNTHETIC UNIT FIXTURE\n" + n.SEPARATOR + "\n" + " ".join(columns) + "\n" + " ".join([header_node] * (len(columns) - 1)) + "\n" + stream.getvalue()).encode()


def write_case(root, suffix="control", amplitude=0, text=None, raw=None):
    simulation_id = "pristine__candidate_000__" + suffix
    directory = Path(root) / simulation_id
    directory.mkdir()
    conf = (text or configuration(amplitude)).encode()
    source = raw or native_raw(amplitude)
    (directory / "configuration.conf").write_bytes(conf)
    (directory / "native.output").write_bytes(source)
    return {"simulation_id": simulation_id, "variant": "pristine", "account_class": "combined",
            "status": "finite", "error": "", "pair_id": "candidate_000", "param_gain_factor": "1",
            "param_coupling_factor": "1", "param_history_factor": "1", "input_amplitude": str(amplitude),
            "input_onset_s": "12", "input_width_s": ".02", "source_output_path": str(directory / "native.output"),
            "source_output_sha256": hashlib.sha256(source).hexdigest(), "config_sha256": hashlib.sha256(conf).hexdigest()}


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        fields = sorted(set().union(*(r.keys() for r in rows)))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"path": str(path), "sha256": n.sha256(path)}


class NativePopulationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.root = Path(self.temporary.name)
        self.environment = patch.dict(os.environ, {"SLURM_JOB_ID": "12345"})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        assert self.root.resolve().parent == Path(__file__).resolve().parent
        self.temporary.cleanup()

    def test_all_nine_populations_and_inputs_with_configured_connections(self):
        row = write_case(self.root)
        trace = n.load_native(row, self.root)
        self.assertEqual(trace["values"].shape, (5000, 18))
        self.assertEqual(trace["V"].shape, (5000, 9))
        self.assertTrue(np.array_equal(trace["V"], trace["voltage"]))
        self.assertEqual(trace["config"]["spatial_nodes"], 1)
        self.assertEqual(trace["config"]["connection_count"], 41)
        self.assertGreater(trace["config"]["external_input_connection_count"], 0)
        self.assertLess(trace["config"]["endogenous_connection_count"], 41)

    def test_sha_change_is_fatal_provenance_error(self):
        row = write_case(self.root)
        row["source_output_sha256"] = "0" * 64
        with self.assertRaises(n.ProvenanceError):
            n.load_native(row, self.root)

    def test_config_sha_and_variant_are_bound(self):
        row = write_case(self.root)
        bad = row | {"config_sha256": "0" * 64}
        with self.assertRaises(n.ProvenanceError):
            n.load_native(bad, self.root)
        with self.assertRaises(n.ProvenanceError):
            n.load_native(row | {"variant": "scaled_dendrite_v1_explicit_patch"}, self.root)

    def test_path_cannot_leave_bank_or_change_simulation_identity(self):
        row = write_case(self.root)
        with self.assertRaises(n.ProvenanceError):
            n.load_native(row, self.root / "other")
        with self.assertRaises(n.ProvenanceError):
            n.load_native(row | {"simulation_id": "pristine__another"}, self.root)

    def test_malformed_clock_fails_without_resampling(self):
        raw = native_raw(mutate=lambda x, c: x.__setitem__((4, 0), .0081))
        with self.assertRaisesRegex(ValueError, "fixed 2-ms"):
            n.parse_native_bytes(raw)

    def test_actual_native_clock_labels_are_preserved(self):
        parsed = n.parse_native_bytes(native_raw())
        np.testing.assert_allclose(parsed["times"][[0, 1, -1]], [.002, .004, 10], rtol=0, atol=1e-12)
        self.assertEqual(parsed["times"][999], 2)

    def test_zero_origin_clock_is_rejected_instead_of_rebased(self):
        raw = native_raw(mutate=lambda x, c: x.__setitem__((slice(None), 0), np.arange(n.N_ROWS) * n.DT))
        with self.assertRaisesRegex(ValueError, r"preserve \.002\.\.10"):
            n.parse_native_bytes(raw)

    def test_missing_population_not_silently_replaced(self):
        columns = ["Time", *n.COLUMNS[:-1], "Pop.10.Q", "Pop.11.Q"]
        with self.assertRaisesRegex(ValueError, "All nine"):
            n.parse_native_bytes(native_raw(columns=columns))

    def test_nonfinite_and_multinode_are_rejected(self):
        raw = native_raw(mutate=lambda x, c: x.__setitem__((1, 1), np.nan))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            n.parse_native_bytes(raw)
        with self.assertRaisesRegex(ValueError, "one node"):
            n.parse_native_bytes(native_raw(header_node="2"))

    def test_separator_is_an_exact_line_not_regex_substring(self):
        raw = native_raw().replace(n.SEPARATOR.encode(), ("prefix" + n.SEPARATOR).encode())
        with self.assertRaisesRegex(ValueError, "separator"):
            n.parse_native_bytes(raw)

    def pair(self, seed=123, gain=1):
        control = n.load_native(write_case(self.root), self.root)
        pulse = n.load_native(write_case(self.root, "pulse", 10, text=configuration(10, gain, seed)), self.root)
        return control, pulse

    def test_actual_pulse_configuration_noise_and_baseline_qualified(self):
        control, pulse = self.pair()
        proof = n.qualify_pair(control, pulse)
        self.assertEqual(proof["onset_relative_s"], 2)
        self.assertEqual(proof["amplitude"], 10)
        self.assertTrue(proof["full_noise_stream_exactly_equal"])

    def test_pulse_onset_stays_two_seconds_on_actual_native_clock(self):
        control, pulse = self.pair()
        proof = n.qualify_pair(control, pulse)
        self.assertEqual(proof["onset_relative_s"], 2)
        self.assertEqual(pulse["times"][0], .002)
        self.assertEqual(pulse["times"][999], 2)
        self.assertEqual(pulse["stimulus"][998], 0)
        self.assertEqual(pulse["stimulus"][999], 10)
        self.assertEqual(pulse["times"][-1], 10)

    def test_seed_metadata_change_requires_entire_observed_stream_match(self):
        control, pulse = self.pair(seed=456)
        self.assertTrue(n.qualify_pair(control, pulse)["WhiteSeed_metadata_changed"])
        pulse["noise"][-1] += 1
        with self.assertRaisesRegex(ValueError, "noise streams"):
            n.qualify_pair(control, pulse)

    def test_hidden_parameter_change_cannot_be_input_response(self):
        control, pulse = self.pair(gain=2)
        with self.assertRaisesRegex(ValueError, "configurations differ"):
            n.qualify_pair(control, pulse)

    def test_all_nine_population_baseline_is_required(self):
        control, pulse = self.pair()
        pulse["values"][0, -1] += .01
        with self.assertRaisesRegex(ValueError, "before the input"):
            n.qualify_pair(control, pulse)

    def test_unobserved_or_wrong_input_cannot_be_qualified(self):
        control, pulse = self.pair()
        pulse["stimulus_observed"] = False
        with self.assertRaisesRegex(ValueError, "physically observed"):
            n.qualify_pair(control, pulse)
        pulse["stimulus_observed"] = True
        pulse["stimulus"][:] = 0
        with self.assertRaisesRegex(ValueError, "amplitude differs"):
            n.qualify_pair(control, pulse)

    def test_cache_roundtrip_preserves_failed_rows_and_full_input_arrays(self):
        row = write_case(self.root)
        failed = row | {"simulation_id": "pristine__failed", "status": "native_failed", "error": "synthetic explicit failure", "source_output_path": ""}
        spec = write_csv(self.root / "inventory.csv", [row, failed])
        inputs = {"bank_inventory": spec, "bank_root": str(self.root)}
        bank = n.load_bank(inputs, self.root / "cache")
        again = n.read_cache(self.root / "cache", expected_inventory_sha256=spec["sha256"])
        self.assertEqual(len(again["rows"]), 2)
        self.assertEqual(again["receipt"]["dispositions"]["retained_native_failure"], 1)
        self.assertEqual(again["receipt"]["output_first_time_s"], .002)
        self.assertEqual(again["receipt"]["output_last_time_s"], 10)
        self.assertTrue(again["receipt"]["output_time_labels_preserved"])
        self.assertFalse(again["receipt"]["output_clock_rebased"])
        simulation_id = row["simulation_id"]
        for field in ("values", "noise", "stimulus", "times"):
            np.testing.assert_array_equal(bank["traces"][simulation_id][field], again["traces"][simulation_id][field])
        self.assertEqual(len(list((self.root / "cache").iterdir())), 4)

    def test_cache_tampering_is_not_an_exclusion(self):
        row = write_case(self.root)
        spec = write_csv(self.root / "inventory.csv", [row])
        n.load_bank({"bank_inventory": spec}, self.root / "cache")
        with (self.root / "cache" / "population_metadata.json").open("a") as stream:
            stream.write(" ")
        with self.assertRaises(n.ProvenanceError):
            n.read_cache(self.root / "cache")

    def test_cache_wrong_zero_origin_clock_rejected_even_with_matching_hash(self):
        row = write_case(self.root)
        spec = write_csv(self.root / "inventory.csv", [row])
        directory = self.root / "cache"
        n.load_bank({"bank_inventory": spec}, directory)
        path = directory / "population_cache.npz"
        with np.load(path, allow_pickle=False) as arrays:
            changed = {key: arrays[key] for key in arrays.files}
        changed["times"] = np.arange(n.N_ROWS) * n.DT
        with path.open("wb") as stream:
            np.savez_compressed(stream, **changed)
        receipt_path = directory / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["files"][path.name] = {"sha256": n.sha256(path), "bytes": path.stat().st_size}
        receipt_path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(n.ProvenanceError, "invalid numeric"):
            n.read_cache(directory)

    def test_cache_clock_receipt_contract_required(self):
        row = write_case(self.root)
        spec = write_csv(self.root / "inventory.csv", [row])
        directory = self.root / "cache"
        n.load_bank({"bank_inventory": spec}, directory)
        receipt_path = directory / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["output_first_time_s"] = 0
        receipt_path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(n.ProvenanceError, "clock metadata"):
            n.read_cache(directory)

    def test_invalid_finite_row_gets_explicit_disposition(self):
        row = write_case(self.root)
        invalid = write_case(self.root, "invalid", raw=native_raw(mutate=lambda x, c: x.__setitem__((0, 1), np.inf)))
        spec = write_csv(self.root / "inventory.csv", [row, invalid])
        bank = n.load_bank({"bank_inventory": spec}, self.root / "cache")
        self.assertEqual(bank["receipt"]["qualified_rows"], 1)
        self.assertEqual(bank["receipt"]["dispositions"]["excluded_numeric_or_schema"], 1)

    def test_required_invalid_row_stops_analysis(self):
        row = write_case(self.root)
        invalid = write_case(self.root, "invalid", raw=native_raw(mutate=lambda x, c: x.__setitem__((0, 1), np.inf)))
        spec = write_csv(self.root / "inventory.csv", [row, invalid])
        with self.assertRaisesRegex(ValueError, "did not qualify"):
            n.load_bank({"bank_inventory": spec, "required_simulation_ids": [invalid["simulation_id"]]}, self.root / "cache")
        with (self.root / "cache" / "population_dispositions.csv").open(newline="", encoding="utf-8") as stream:
            dispositions = list(csv.DictReader(stream))
        self.assertEqual(len(dispositions), 2)
        rejected = next(x for x in dispositions if x["simulation_id"] == invalid["simulation_id"])
        self.assertEqual(rejected["population_status"], "excluded_numeric_or_schema")
        self.assertIn("nonfinite", rejected["population_error"])
        receipt = json.loads((self.root / "cache" / "receipt.json").read_text())
        self.assertEqual(receipt["status"], "native_population_cache_rejected")
        self.assertEqual(receipt["missing_required_simulation_ids"], [invalid["simulation_id"]])
        self.assertIn("nonfinite", receipt["rejection_representatives"]["groups"][0]["population_error"])
        before = {p.name: p.read_bytes() for p in (self.root / "cache").iterdir()}
        with self.assertRaisesRegex(n.ProvenanceError, "Unqualified"):
            n.load_bank({"bank_inventory": spec}, self.root / "cache")
        self.assertEqual(before, {p.name: p.read_bytes() for p in (self.root / "cache").iterdir()})

    def test_all_rows_rejected_still_preserves_reasons(self):
        invalid = write_case(self.root, raw=native_raw(mutate=lambda x, c: x.__setitem__((slice(None), 0), np.arange(n.N_ROWS) * n.DT)))
        spec = write_csv(self.root / "inventory.csv", [invalid])
        with self.assertRaisesRegex(ValueError, "did not qualify"):
            n.load_bank({"bank_inventory": spec}, self.root / "cache")
        receipt = json.loads((self.root / "cache" / "receipt.json").read_text())
        self.assertEqual(receipt["qualified_rows"], 0)
        self.assertEqual(receipt["dispositions"]["excluded_numeric_or_schema"], 1)
        self.assertIn(".002..10", receipt["rejection_representatives"]["groups"][0]["population_error"])
        self.assertEqual(set(p.name for p in (self.root / "cache").iterdir()), {"receipt.json", "population_dispositions.csv"})

    def test_error_representatives_are_bounded_without_losing_csv_reasons(self):
        rows = [{"simulation_id": f"synthetic_{i}_{j}", "population_status": "excluded_numeric_or_schema",
                 "population_error": f"{i}:" + "x" * 1000} for i in range(25) for j in range(10)]
        summary = n._error_representatives(rows)
        self.assertEqual(summary["distinct_error_groups"], 25)
        self.assertEqual(len(summary["groups"]), 20)
        self.assertTrue(all(len(x["simulation_ids"]) == 8 and len(x["population_error"]) == 512
                            and x["count"] == 10 for x in summary["groups"]))

    def test_inventory_and_parent_receipt_identity_are_bound(self):
        row = write_case(self.root)
        spec = write_csv(self.root / "inventory.csv", [row])
        parent = self.root / "bank_receipt.json"
        parent.write_text(json.dumps({"status": "native_simulation_bank_completed", "inventory_sha256": "0" * 64, "candidate_count": 1, "variants": [{"variant": "pristine"}]}))
        with self.assertRaises(n.ProvenanceError):
            n.load_bank({"bank_inventory": spec, "bank_receipt": {"path": str(parent), "sha256": n.sha256(parent)}}, self.root / "cache")

    def test_scored_table_cannot_change_configuration_identity(self):
        row = write_case(self.root)
        spec = write_csv(self.root / "inventory.csv", [row])
        scored = write_csv(self.root / "scored.csv", [row | {"config_sha256": "0" * 64, "clinical_score": ".4"}])
        with self.assertRaises(n.ProvenanceError):
            n.load_bank({"bank_inventory": spec, "scored_bank": scored}, self.root / "cache")

    def test_full_bank_cannot_read_on_login_without_slurm(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": ""}):
            with self.assertRaisesRegex(RuntimeError, "scheduled"):
                n.load_bank({}, self.root / "cache")


if __name__ == "__main__":
    unittest.main()
