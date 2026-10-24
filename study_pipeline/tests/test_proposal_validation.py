"""Counterexamples for independent sampling and supplied human qualification.

All files are temporary synthetic test fixtures. They never enter a study run
or constitute actual human qualification evidence.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from adversarial_study import core, proposal_validation as validation


class ProspectiveFixtures:
    def __init__(self, root, kind="replication", mode="confirmatory", n=2):
        self.root = Path(root)
        self.kind = kind
        self.now = datetime.now(timezone.utc)
        self.code = self.root / "synthetic_fixture_source.txt"
        self.code.write_text("Synthetic test fixture only; no real human evidence", encoding="utf-8")
        self.pin = validation._pin(self.code)
        config = core.read_json(validation.DEFAULT_CONFIG)
        config["bootstrap_samples"] = 100
        self.config = self.json("config.json", config)
        people = ["d1", "d2", "m1", "m2", "m3", "m4"]
        self.development = self.csv("development.csv", [{"participant_id": p, "identity_fingerprint": "global-"+p} for p in people])
        self.replication = self.csv("replication.csv", [{"participant_id": "r1", "identity_fingerprint": "global-r1"}])
        simulations = self.csv("precision.csv", [{"scenario_id": s, "informative_people": n, "interval_width_95": .5, "informative_yield": .5, "missing_outcome_rate": .1} for s in ("null", "adverse_missingness")])
        precision = self.json("precision.json", {"status": "finalized_before_outcomes", "analysis_mode": mode, "informative_people": n, "eligible_recruitment_target": 2*n, "maximum_interval_width": 1., "reliability_definition": "Test fixture precision only", "episode_schedule": "Two original condition observations per person", "required_scenarios": ["null", "adverse_missingness"], "simulation_table": simulations, "simulation_provenance": [self.pin]})
        raw = self.csv("raw_calibration.csv", [{"validation_id": p, "participant_id": p, "x": .3} for p in ("d1", "d2")])
        endpoint = self.csv("endpoint_calibration.csv", [{"validation_id": p, "participant_id": p, "endpoint_value": .3} for p in ("d1", "d2")])
        qa = self.json("endpoint_qa.json", {"assay_id": "fixture_assay", "unit": "normalized", "minimum_validation_people": 2, "raw_calibration_table": raw, "validation_table": endpoint, "raw_source_artifacts": [self.pin]})
        self.spec = {"schema_version": 1, "kind": kind, "analysis_mode": mode, "configuration": self.config,
                     "computational_candidate": self.pin, "development_registry": self.development, "replication_registry": self.replication if kind == "experiment" else None,
                     "precision_plan": precision, "account_predictions": None, "qualification": {"endpoint_validation": qa},
                     "endpoint": {"name": "fixture_endpoint", "unit": "normalized", "source_column": "x", "operator_id": "identity_fixture", "construct": "Synthetic fixture, never human evidence", "report_timing_basis": "original timed fixture", "minimum": 0., "maximum": 1., "scale": 1., "offset": 0., "validation_absolute_tolerance": 1e-8, "operator_source": self.pin},
                     "sampling": {"conditions": ["target", "comparator"] if kind == "replication" else ["active", "control"], "assay_id": "fixture_assay", "minimum_sources": 2 if kind == "replication" else 1, "minimum_sites": 1 if kind == "replication" else 2, "minimum_sessions_per_person": 1 if kind == "replication" else 2, "minimum_baseline_observations": 1, "eligibility_rule": {"allowed_pre_window_quality": ["qualified"], "window_seconds": 10., "clinical_score_max_for_enrichment": 8.}, "condition_rules": {c: [{"field": "condition_code", "operator": "equal", "value": c}] for c in ("target", "comparator")} if kind == "replication" else None, "sampling_rule": "fixed fixture observations", "report_schedule": "original fixture clocks"},
                     "informative": {"minimum_target_or_active_observations": 1, "minimum_comparator_or_control_observations": 1, "maximum_missing_fraction": 1.}}
        if kind == "experiment":
            self.spec["sampling"]["baseline_analysis"] = {"low_clinical_score_max": 8., "reported_experience_categories": ["reported_experience"], "minimum_repeat_pairs_per_person": 1}
            self.experiment_qualification()
        self.register()

    def json(self, name, value):
        path = self.root / name
        core.write_json(path, value)
        return validation._pin(path)

    def csv(self, name, rows, fields=None):
        path = self.root / name
        core.write_csv(path, rows, fields)
        return validation._pin(path)

    def register(self):
        self.spec["registration"] = self.json("registration.json", {"analysis_contract_sha256": validation._canonical(validation.contract(self.spec)), "registered_utc": (self.now-timedelta(days=1)).isoformat(), "registry_identifier": "synthetic-fixture-not-a-real-registration"})
        self.spec_path = self.root / "spec.json"
        core.write_json(self.spec_path, self.spec)

    def experiment_qualification(self):
        predictions = self.csv("predictions.csv", [{"prediction_id": a+":"+c, "account_id": a, "condition": c, "predicted_endpoint": .3, "unit": "normalized", "assay_id": "fixture_assay", "input_variable": "synthetic_fixture_only", "input_unit": "fixture_stimulus", "intended_input": 1. if c == "active" else 0., "native_input": 1. if c == "active" else 0., "native_input_variable": "fixture_native_input", "native_input_unit": "fixture_native_unit"} for a in ("A", "B") for c in ("active", "control")])
        self.spec["account_predictions"] = predictions
        rawmap = self.csv("mapping_raw.csv", [{"observation_id": p+":mapping", "participant_id": p, "split": s, "observed": .3} for p, s in (("m1", "train"), ("m2", "validation"), ("m3", "test"), ("m4", "test"))])
        maptable = self.csv("mapping_predictions.csv", [{**r, "predicted": .3} for r in core.read_csv(rawmap["path"])])
        mapping = self.json("mapping.json", {"assay_id": "fixture_assay", "unit": "normalized", "minimum_test_people": 2, "predeclared_maximum_mse": .01, "validation_table": maptable, "operator_source": self.pin, "raw_observation_table": rawmap, "frozen_mapping": self.pin})
        human_training = self.csv("human_training_observations.csv", [{"observation_id": "m1:mapping", "participant_id": "m1", "split": "train", "observed": "0.3"}, {"observation_id": "m2:mapping", "participant_id": "m2", "split": "validation", "observed": "0.3"}])
        calibrated_inputs = [{"calibration_id": c, "condition": c, "model_input": v, "measured_physical_input": v} for c, v in (("active", 1.), ("control", 0.))]
        calibration = self.csv("input_mapping_calibration.csv", calibrated_inputs)
        raw_inputs = self.csv("raw_input_mapping.csv", [{k: row[k] for k in ("calibration_id", "condition", "measured_physical_input")} for row in calibrated_inputs])
        input_mapping = self.json("input_mapping.json", {"assay_id": "fixture_assay", "physical_variable": "synthetic_fixture_only", "physical_unit": "fixture_stimulus", "model_variable": "fixture_native_input", "model_unit": "fixture_native_unit", "scale": 1., "offset": 0., "predeclared_absolute_mapping_tolerance": .01, "calibration_table": calibration, "raw_human_input_table": raw_inputs, "mapping_operator_source": self.pin, "native_input_execution": self.pin, "raw_physical_source": self.pin})
        artifacts = {}
        for a in ("A", "B"):
            params = self.root / (a+"_params.txt")
            params.write_text(a+" fixture adapted params", encoding="utf-8")
            pin = validation._pin(params)
            training = self.json(a+"_training.json", {"assay_id": "fixture_assay", "adaptation_kind": "neural_optimizer", "actual_optimizer_steps": 2, "original_weights_sha256": self.pin["sha256"], "adapted_weights_sha256": pin["sha256"], "adapted_parameters_sha256": pin["sha256"], "human_training_observations": human_training, "split": {"train_participant_ids": ["m1"], "validation_participant_ids": ["m2"], "test_participant_ids": ["m3", "m4"]}})
            artifacts[a] = {"source_code": self.pin, "adapted_parameters": pin, "training_receipt": training, "human_assay_mapping": mapping, "physical_input_mapping": input_mapping}
        self.spec["qualification"]["account_execution"] = self.json("account_execution.json", {"assay_id": "fixture_assay", "unit": "normalized", "prediction_table_sha256": predictions["sha256"], "native_execution_table": predictions, "account_artifacts": artifacts})
        actual = self.csv("actual_delivery.csv", [{"calibration_id": c, "condition": c, "delivered": v, "unit": "fixture_stimulus", "delivered_utc": (self.now-timedelta(days=2)).isoformat()} for c, v in (("active", 1.), ("control", 0.))])
        prescribed = self.csv("intended_delivery.csv", [{"calibration_id": c, "condition": c, "intended": v, "unit": "fixture_stimulus"} for c, v in (("active", 1.), ("control", 0.))])
        self.spec["qualification"]["physical_intervention"] = self.json("physical_qa.json", {"assay_id": "fixture_assay", "physical_variable": "synthetic_fixture_only", "unit": "fixture_stimulus", "predeclared_absolute_delivery_tolerance": .001, "raw_delivery_table": actual, "prescribed_delivery_table": prescribed, **{k: self.pin for k in ("delivery_operator_source", "approved_acquisition_protocol", "ethics_authorization", "raw_delivery_source")}})
        self.spec["randomization_plan"] = self.json("randomization.json", {"method": "fixed_block_randomization", "allocation_generator": self.pin, "strata": [{"stratum_id": "stratum", "block_size": 2, "active_per_block": 1, "probability_active": .5}]})

    def freeze(self):
        self.frozen = self.root / "frozen"
        return validation.freeze(self.spec_path, self.frozen)

    def outcomes(self, values=(.3, .3), people=("new1", "new2")):
        date = self.now+timedelta(days=2)
        participants, rows = [], []
        for i, p in enumerate(people):
            source, site = "source"+str(i % 2), "site"+str(i % 2)
            participants.append({"participant_id": p, "identity_fingerprint": "global-"+p, "source_id": source, "site_id": site, "eligible": "1", "eligibility_reason": "prospective eligibility", "detector_exposure": "no_known_exposure", "analyst_exposure": "unopened_before_freeze"})
            for j, condition in enumerate(self.spec["sampling"]["conditions"]):
                start = date+timedelta(minutes=i*10+j)
                row = {"observation_id": p+":"+str(j), "participant_id": p, "session_id": "diagnostic" if self.kind == "experiment" else "session1", "source_id": source, "site_id": site, "condition": condition, "sampling_mode": "random_probe", "eligible": "1", "eligibility_reason": "qualified pre-window", "pre_clinical_score": "5", "pre_window_quality": "qualified", "eligibility_window_start_utc": start.isoformat(), "eligibility_window_end_utc": (start+timedelta(seconds=10)).isoformat(), "eligibility_decided_utc": (start+timedelta(seconds=11)).isoformat(), "outcome_acquired_utc": (start+timedelta(seconds=30)).isoformat(), "endpoint_value": str(values[j]) if values[j] is not None else "", "missing_reason": "equipment_failure" if values[j] is None else ""}
                if self.kind == "experiment":
                    row.update(block_id="block", stratum_id="stratum", assignment_probability_active="0.5", randomized_utc=(start+timedelta(seconds=12)).isoformat(), intervention_start_utc=(start+timedelta(seconds=13)).isoformat(), clinical_window_start_utc=(start+timedelta(seconds=14)).isoformat(), clinical_window_end_utc=(start+timedelta(seconds=24)).isoformat(), response_window_start_utc=(start+timedelta(seconds=13)).isoformat(), response_window_end_utc=(start+timedelta(seconds=14)).isoformat(), experience_report_utc=(start+timedelta(seconds=25)).isoformat(), clinical_score="5", experience_report="reported_experience", experience_missing_reason="", capacity_name="fixture_detection", capacity_value="1", capacity_missing_reason="", physiology_name="", physiology_value="", physiology_missing_reason="not_acquired")
                rows.append(row)
        self.data = {"frozen_protocol_sha256": core.sha256_file(self.frozen / "frozen_protocol.json"), "human_data": True, "synthetic": False, "participant_registry": self.csv("participants.csv", participants), "raw_source_artifacts": [self.pin]}
        self.rows = rows
        self.refresh_outcomes()
        if self.kind == "experiment":
            sessions, baseline = [], []
            for i, person in enumerate(participants):
                for role, begin, end in (("baseline", date-timedelta(days=1), date-timedelta(days=1)+timedelta(hours=1)), ("diagnostic", date-timedelta(minutes=1), date+timedelta(hours=1))):
                    sessions.append({k: person[k] for k in ("participant_id", "source_id", "site_id")} | {"session_id": role, "session_role": role, "acquisition_start_utc": begin.isoformat(), "acquisition_end_utc": end.isoformat(), "qualified_baseline_observations": 2 if role == "baseline" else 0, "raw_session_sha256": self.pin["sha256"]})
                for episode in range(2):
                    baseline.append({"observation_id": person["participant_id"]+":baseline:"+str(episode), "participant_id": person["participant_id"], "session_id": "baseline", "clinical_score": "5", "experience_report": "reported_experience", "experience_missing_reason": "", "capacity_name": "fixture_detection", "capacity_value": "1", "capacity_missing_reason": "", "acquired_utc": (date-timedelta(days=1)+timedelta(minutes=5+episode)).isoformat()})
            self.data.update(session_registry=self.csv("sessions.csv", sessions), raw_session_artifacts=[self.pin], baseline_observations=self.csv("baseline.csv", baseline))
            from adversarial_study.native import CHECKPOINT_SHA256
            self.data["clinical_scoring_receipt"] = self.json("clinical_scoring.json", {"original_checkpoint_sha256": CHECKPOINT_SHA256, "scored_observations_sha256": self.data["raw_endpoint_table"]["sha256"], "baseline_observations_sha256": self.data["baseline_observations"]["sha256"], "source_artifacts": [self.pin]})
            self.data["first_outcome_acquired_utc"] = min(r["acquired_utc"] for r in baseline)
        self.data_path = self.root / "data.json"
        core.write_json(self.data_path, self.data)
        return rows

    def refresh_outcomes(self):
        self.data["observations"] = self.csv("outcomes.csv", self.rows)
        self.data["acquisition_registry"] = self.csv("acquisition.csv", [{k: v for k, v in r.items() if k not in validation.OUTCOME_FIELDS} for r in self.rows])
        self.data["raw_endpoint_table"] = self.csv("raw_endpoints.csv", [{**r, "x": r["endpoint_value"], "condition_code": r["condition"]} for r in self.rows])
        self.data["first_outcome_acquired_utc"] = min(r["outcome_acquired_utc"] for r in self.rows)

    def save(self):
        core.write_json(self.data_path, self.data)

    def evaluate(self):
        return validation.evaluate(self.frozen, self.data_path, self.root / "evaluation", kind=self.kind)


class ProposalValidationTests(unittest.TestCase):
    def fixture(self, directory, **kwargs):
        return ProspectiveFixtures(directory, **kwargs)

    def test_templates_are_inadmissible_without_real_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d)/"templates"
            validation.prepare_template("experiment", target)
            with self.assertRaises((core.StudyError, OSError)):
                validation.freeze(target/"specification.template.json", Path(d)/"frozen")
            self.assertFalse(core.read_json(target/"template_status.json")["human_intervention_qualified"])

    def test_independent_null_replication_completes_and_uses_people(self):
        with tempfile.TemporaryDirectory() as d:
            f = self.fixture(d); f.freeze(); f.outcomes()
            result = f.evaluate()
            self.assertEqual(result["point_estimate"], 0.)
            self.assertTrue(result["scientific_completion"])
            self.assertEqual(result["census"]["acquired_people"], 2)

    def test_negative_replication_is_retained(self):
        with tempfile.TemporaryDirectory() as d:
            f = self.fixture(d); f.freeze(); f.outcomes((.2, .8))
            result = f.evaluate()
            self.assertAlmostEqual(result["point_estimate"], -.6)
            self.assertTrue(result["scientific_completion"])

    def test_unequal_episode_counts_do_not_change_independent_person_weights(self):
        spec = {"kind": "replication", "sampling": {"conditions": ["target", "comparator"], "minimum_sessions_per_person": 1}, "endpoint": {"minimum": 0., "maximum": 1.}, "informative": {"minimum_target_or_active_observations": 1, "minimum_comparator_or_control_observations": 1, "maximum_missing_fraction": .2}}
        config = {"seed": 20261008, "bootstrap_samples": 100}
        people = {p: {"eligible": "1", "source_id": "source", "site_id": "site"} for p in ("many", "few")}
        rows = [{"participant_id": "many", "session_id": "s", "condition": c, "eligible": "1", "endpoint_value": str(v)} for _ in range(20) for c, v in (("target", 1.), ("comparator", 0.))]
        rows += [{"participant_id": "few", "session_id": "s", "condition": c, "eligible": "1", "endpoint_value": str(v)} for c, v in (("target", 0.), ("comparator", 1.))]
        result = validation.estimate(rows, people, spec, config)
        self.assertEqual(result["point_estimate"], 0.)
        self.assertEqual(result["census"]["informative_people"], 2)

    def test_missing_entire_condition_retains_person_in_conservative_bounds(self):
        spec = {"kind": "replication", "sampling": {"conditions": ["target", "comparator"], "minimum_sessions_per_person": 1}, "endpoint": {"minimum": 0., "maximum": 1.}, "informative": {"minimum_target_or_active_observations": 1, "minimum_comparator_or_control_observations": 1, "maximum_missing_fraction": .2}}
        people = {p: {"eligible": "1", "source_id": "source", "site_id": "site"} for p in ("complete", "missing")}
        rows = [{"participant_id": "complete", "session_id": "s", "condition": c, "eligible": "1", "endpoint_value": ".3"} for c in ("target", "comparator")]
        result = validation.estimate(rows, people, spec, {"seed": 20261008, "bootstrap_samples": 100})
        self.assertIsNone(result["point_estimate"])
        self.assertEqual(result["census"]["eligible_people"], 2)
        self.assertEqual(result["identification_interval"], [-.5, .5])

    def test_pilot_never_completes_confirmatory_programme(self):
        with tempfile.TemporaryDirectory() as d:
            f = self.fixture(d, mode="pilot"); f.freeze(); f.outcomes()
            result = f.evaluate()
            self.assertFalse(result["scientific_completion"])
            self.assertIn("pilot_cannot_complete_confirmatory_programme", result["incomplete_reasons"])

    def test_missing_outcomes_keep_complete_people_and_bounds(self):
        with tempfile.TemporaryDirectory() as d:
            f = self.fixture(d); f.freeze(); f.outcomes((None, .2))
            result = f.evaluate()
            self.assertIsNone(result["point_estimate"])
            self.assertEqual(result["census"]["missing_outcomes"], 2)
            self.assertEqual(result["census"]["eligible_people"], 2)
            self.assertAlmostEqual(result["identification_interval"][0], -.2)
            self.assertAlmostEqual(result["identification_interval"][1], .8)

    def test_omitted_original_acquisition_fails_even_when_outcomes_rehashed(self):
        with tempfile.TemporaryDirectory() as d:
            f = self.fixture(d); f.freeze(); f.outcomes()
            f.data["observations"] = f.csv("outcomes.csv", f.rows[:-1]); f.save()
            with self.assertRaisesRegex(core.StudyError, "Every original"):
                f.evaluate()

    def test_exposure_and_identity_alias_fail_independence(self):
        for fault in ("exposure", "alias", "development"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as d:
                f = self.fixture(d); f.freeze(); f.outcomes()
                people = core.read_csv(f.data["participant_registry"]["path"])
                if fault == "exposure": people[0]["detector_exposure"] = "unknown"
                elif fault == "alias": people[0]["identity_fingerprint"] = people[1]["identity_fingerprint"]
                else: people[0]["identity_fingerprint"] = "global-d1"
                f.data["participant_registry"] = f.csv("participants.csv", people); f.save()
                with self.assertRaises(core.StudyError): f.evaluate()

    def test_hindsight_eligibility_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            f = self.fixture(d); f.freeze(); f.outcomes()
            f.rows[0]["pre_window_quality"] = "artifact"
            f.refresh_outcomes(); f.save()
            with self.assertRaisesRegex(core.StudyError, "Eligibility flag"):
                f.evaluate()

    def test_relabelled_replication_membership_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            f = self.fixture(d); f.freeze(); f.outcomes()
            raw = core.read_csv(f.data["raw_endpoint_table"]["path"])
            raw[0]["condition_code"] = "comparator"
            f.data["raw_endpoint_table"] = f.csv("raw_endpoints.csv", raw); f.save()
            with self.assertRaisesRegex(core.StudyError, "contrast membership"):
                f.evaluate()

    def test_source_operator_correspondence_is_rechecked(self):
        with tempfile.TemporaryDirectory() as d:
            f = self.fixture(d); f.freeze(); f.outcomes()
            raw = core.read_csv(f.data["raw_endpoint_table"]["path"])
            raw[0]["x"] = ".9"
            f.data["raw_endpoint_table"] = f.csv("raw_endpoints.csv", raw); f.save()
            with self.assertRaisesRegex(core.StudyError, "frozen source operator"):
                f.evaluate()

    def test_two_sessions_with_baseline_then_randomized_diagnostic_complete(self):
        with tempfile.TemporaryDirectory() as d:
            f = self.fixture(d, kind="experiment"); f.freeze(); f.outcomes((.8, .2))
            result = f.evaluate()
            self.assertAlmostEqual(result["point_estimate"], .6)
            self.assertTrue(result["scientific_completion"])
            self.assertTrue(all(r["sessions"] == 2 and r["baseline_acquisition_complete"] for r in result["participant_estimates"]))
            self.assertTrue(result["acquisition_protocol_completion"])
            self.assertFalse(result["mechanism_discrimination_design_qualified"])
            self.assertFalse(result["mechanism_discriminating_test_completed"])
            self.assertEqual(result["baseline_occurrence_repeatability"]["sufficient_repeatability_people"], 2)

    def test_predicted_account_separation_does_not_require_positive_observed_effect(self):
        with tempfile.TemporaryDirectory() as d:
            f = self.fixture(d, kind="experiment")
            predictions = core.read_csv(f.spec["account_predictions"]["path"])
            for row in predictions:
                row["predicted_endpoint"] = str(.9 if (row["account_id"] == "A") == (row["condition"] == "active") else .1)
            f.spec["account_predictions"] = f.csv("predictions.csv", predictions)
            execution = core.read_json(f.spec["qualification"]["account_execution"]["path"])
            execution["prediction_table_sha256"] = f.spec["account_predictions"]["sha256"]
            execution["native_execution_table"] = f.spec["account_predictions"]
            f.spec["qualification"]["account_execution"] = f.json("account_execution.json", execution); f.register()
            f.freeze(); f.outcomes()
            result = f.evaluate()
            self.assertTrue(result["mechanism_discrimination_design_qualified"])
            self.assertEqual(result["point_estimate"], 0.)
            self.assertTrue(result["mechanism_discriminating_test_completed"])

    def test_missing_baseline_reports_produce_bounds_and_incomplete_repeatability(self):
        with tempfile.TemporaryDirectory() as d:
            f = self.fixture(d, kind="experiment"); f.freeze(); f.outcomes()
            baseline = core.read_csv(f.data["baseline_observations"]["path"])
            for row in baseline:
                row["experience_report"] = ""; row["experience_missing_reason"] = "no_original_report"
            f.data["baseline_observations"] = f.csv("baseline.csv", baseline)
            scoring = core.read_json(f.data["clinical_scoring_receipt"]["path"])
            scoring["baseline_observations_sha256"] = f.data["baseline_observations"]["sha256"]
            f.data["clinical_scoring_receipt"] = f.json("clinical_scoring.json", scoring); f.save()
            result = f.evaluate()
            self.assertFalse(result["scientific_completion"])
            summary = result["baseline_occurrence_repeatability"]
            self.assertEqual(summary["sufficient_repeatability_people"], 0)
            self.assertEqual(summary["aggregate"]["co_report_prevalence_lower"]["equal_person_estimate"], 0.)
            self.assertEqual(summary["aggregate"]["co_report_prevalence_upper"]["equal_person_estimate"], 1.)

    def test_experiment_excludes_both_replication_and_development_people(self):
        with tempfile.TemporaryDirectory() as d:
            f = self.fixture(d, kind="experiment"); f.freeze(); f.outcomes(people=("r1", "new2"))
            with self.assertRaisesRegex(core.StudyError, "overlap"):
                f.evaluate()

    def test_manual_intervention_boolean_without_calibration_cannot_freeze(self):
        with tempfile.TemporaryDirectory() as d:
            f = self.fixture(d, kind="experiment")
            f.spec["qualification"]["physical_intervention"] = f.json("manual_qa.json", {"human_intervention_qualified": True})
            f.register()
            with self.assertRaises(core.StudyError): f.freeze()

    def test_mapping_test_cannot_contain_actual_adaptation_training_people(self):
        with tempfile.TemporaryDirectory() as d:
            f = self.fixture(d, kind="experiment")
            execution = core.read_json(f.spec["qualification"]["account_execution"]["path"])
            account = execution["account_artifacts"]["A"]
            training = core.read_json(account["training_receipt"]["path"])
            actual = core.read_csv(training["human_training_observations"]["path"])
            actual[0]["participant_id"] = "m3"
            training["human_training_observations"] = f.csv("leaking_adaptation.csv", actual)
            training["split"]["train_participant_ids"] = ["m3"]
            account["training_receipt"] = f.json("A_training.json", training)
            f.spec["qualification"]["account_execution"] = f.json("account_execution.json", execution)
            f.register()
            with self.assertRaisesRegex(core.StudyError, "overlap actual human adaptation"):
                f.freeze()

    def test_model_prediction_cannot_change_physical_input_units(self):
        with tempfile.TemporaryDirectory() as d:
            f = self.fixture(d, kind="experiment")
            predictions = core.read_csv(f.spec["account_predictions"]["path"])
            predictions[0]["input_unit"] = "unqualified_native_simulator_units"
            f.spec["account_predictions"] = f.csv("predictions.csv", predictions)
            execution = core.read_json(f.spec["qualification"]["account_execution"]["path"])
            execution["prediction_table_sha256"] = f.spec["account_predictions"]["sha256"]
            execution["native_execution_table"] = f.spec["account_predictions"]
            f.spec["qualification"]["account_execution"] = f.json("account_execution.json", execution); f.register()
            with self.assertRaisesRegex(core.StudyError, "calibrated physical human intervention"):
                f.freeze()

    def test_selected_native_input_must_use_its_verified_physical_mapping(self):
        with tempfile.TemporaryDirectory() as d:
            f = self.fixture(d, kind="experiment")
            predictions = core.read_csv(f.spec["account_predictions"]["path"])
            predictions[0]["native_input"] = "0.5"
            f.spec["account_predictions"] = f.csv("predictions.csv", predictions)
            execution = core.read_json(f.spec["qualification"]["account_execution"]["path"])
            execution["prediction_table_sha256"] = f.spec["account_predictions"]["sha256"]
            execution["native_execution_table"] = f.spec["account_predictions"]
            f.spec["qualification"]["account_execution"] = f.json("account_execution.json", execution); f.register()
            with self.assertRaisesRegex(core.StudyError, "selected native prediction input"):
                f.freeze()

    def test_physical_delivery_errors_cannot_be_qualified_by_rehashing(self):
        with tempfile.TemporaryDirectory() as d:
            f = self.fixture(d, kind="experiment")
            qa = core.read_json(f.spec["qualification"]["physical_intervention"]["path"])
            rows = core.read_csv(qa["raw_delivery_table"]["path"]); rows[0]["delivered"] = "9"
            qa["raw_delivery_table"] = f.csv("actual_delivery.csv", rows)
            f.spec["qualification"]["physical_intervention"] = f.json("physical_qa.json", qa); f.register()
            with self.assertRaisesRegex(core.StudyError, "delivery exceeds"): f.freeze()

    def test_randomization_before_eligibility_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            f = self.fixture(d, kind="experiment"); f.freeze(); f.outcomes()
            f.rows[0]["randomized_utc"] = f.rows[0]["eligibility_window_start_utc"]
            f.refresh_outcomes()
            # Keep genuine baseline's first-outcome time and source scoring binding.
            f.data["first_outcome_acquired_utc"] = min(r["acquired_utc"] for r in core.read_csv(f.data["baseline_observations"]["path"]))
            scoring = core.read_json(f.data["clinical_scoring_receipt"]["path"])
            scoring["scored_observations_sha256"] = f.data["raw_endpoint_table"]["sha256"]
            f.data["clinical_scoring_receipt"] = f.json("clinical_scoring.json", scoring); f.save()
            with self.assertRaisesRegex(core.StudyError, "Eligibility decision"):
                f.evaluate()

    def test_fixed_block_composition_cannot_be_relabelled(self):
        with tempfile.TemporaryDirectory() as d:
            f = self.fixture(d, kind="experiment"); f.freeze(); f.outcomes()
            f.rows[0]["condition"] = "control"; f.refresh_outcomes()
            f.data["first_outcome_acquired_utc"] = min(r["acquired_utc"] for r in core.read_csv(f.data["baseline_observations"]["path"]))
            scoring = core.read_json(f.data["clinical_scoring_receipt"]["path"])
            scoring["scored_observations_sha256"] = f.data["raw_endpoint_table"]["sha256"]
            f.data["clinical_scoring_receipt"] = f.json("clinical_scoring.json", scoring); f.save()
            with self.assertRaisesRegex(core.StudyError, "blocks are missing"):
                f.evaluate()

    def test_final_sample_shortfall_remains_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            f = self.fixture(d, n=4); f.freeze(); f.outcomes()
            result = f.evaluate()
            self.assertFalse(result["scientific_completion"])
            self.assertIn("finalized_informative_sample_not_reached", result["incomplete_reasons"])


if __name__ == "__main__":
    unittest.main()
