"""Tests of diagnostic response behavior, pairing and inference boundaries."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import population_perturbations as p
import native_population as n


def fixture(index=0, amplitude=1., hidden_scale=1., tau=.2):
    t = np.arange(5000) * .002
    base = np.column_stack([.2 * np.sin((j + 1) * t + .1 * j) + .3 * j for j in range(9)])
    rate = 20 + base * 5
    after = t >= 2.
    response = np.zeros(len(t))
    response[after] = np.exp(-(t[after] - 2.) / tau)
    pattern = np.ones(9)
    pattern[1:] *= hidden_scale
    pulse_v = base + response[:, None] * pattern
    pulse_q = rate + response[:, None] * pattern * 4
    noise = np.sin(t * 13.)
    stimulus = np.where((t >= 2.) & (t <= 2.02), amplitude, 0.)
    population_config = '\n'.join(f'Population {j}: Test\n' + ('Stimulus: White - Onset: 0 Mean: 1 StdDev: 2' if j == 10 else
        f'Stimulus: PulseRect - Onset: 12 Amplitude: {{amplitude}} Width: .02 Pulses: 1' if j == 11 else
        'Length: .1\nFiring: Function: Sigmoid Theta: .01 Sigma: .01 Qmax: 100') for j in range(1, 12))
    text = 'Time: 20 Deltat: .0001\nNodes: 1\n' + population_config + '\nOutput: Node: All Start: 10 Interval: .002\nPopulation: 1 2 3 4 5 6 7 8 9 10 11\n'
    meta = dict(account_class='gain', variant='scaled_dendrite_v1_explicit_patch', pair_id=f'candidate_{index:03d}',
                param_gain_factor=1. + index * .01, param_coupling_factor=1., param_history_factor=1.,
                input_onset_s=12., input_width_s=.02, status='finite', score_status='ok', score=.5,
                feature_acf_100ms=.2, feature_acf_500ms=.1, feature_spectral_centroid_hz=10.,
                feature_spectral_entropy=.6, feature_hjorth_mobility=50., feature_std_amplitude=.2)
    traces = []
    for tag, V, Q, amp, stim in [('control', base, rate, 0., np.zeros(len(t))), ('pulse', pulse_v, pulse_q, amplitude, stimulus)]:
        row = meta | dict(simulation_id=f'candidate_{index:03d}_{tag}', input_amplitude=amp)
        values = np.empty((len(t), 18))
        values[:, ::2], values[:, 1::2] = V, Q
        cfg = dict(input_onset_s=12., output_start_s=10., input_width_s=.02, input_amplitude=amp)
        traces.append(dict(row=row, times=t, values=values, voltage=V, rate=Q, noise=noise.copy(), stimulus=stim,
                           stimulus_observed=True, config_text=text.format(amplitude=amp), config=cfg))
    return traces


def bank_fixture(count=2):
    pairs = [fixture(i, hidden_scale=i + 1.) for i in range(count)]
    return {'rows': [v['row'] for pair in pairs for v in pair],
            'traces': {v['row']['simulation_id']: v for pair in pairs for v in pair}}


class ResponseTests(unittest.TestCase):
    def test_exponential_response_has_expected_energy_and_recovery(self):
        c, s = fixture()
        r, delta, scale = p.channel_response(c['voltage'][:, 0], s['voltage'][:, 0], c['times'], 2., .02, population=0, kind='V')
        self.assertAlmostEqual(r['peak_absolute'], 1.)
        self.assertAlmostEqual(r['finite_response_energy'], .1, places=4)
        # exp(-t/.2) reaches five percent at .2 log(20) seconds.
        self.assertAlmostEqual(r['recovery_s'], .2 * np.log(20), delta=.0021)
        self.assertEqual(r['recovery_status'], 'returned_below_5pct_peak_for_250ms')
        self.assertTrue(r['recruited_3sd_for_10ms'])

    def test_persistent_response_is_censored_never_imputed_to_end(self):
        c, s = fixture(tau=1e6)
        r, _, _ = p.channel_response(c['voltage'][:, 0], s['voltage'][:, 0], c['times'], 2., .02, population=0, kind='V')
        self.assertEqual(r['recovery_status'], 'right_censored_at_end')
        self.assertIsNone(r['recovery_s'])
        self.assertAlmostEqual(r['recovery_censoring_time_s'], 7.998)

    def test_zero_response_and_constant_baseline_preserved_without_epsilon_gain(self):
        t = np.arange(5000) * .002
        r, _, scale = p.channel_response(np.ones(5000), np.ones(5000), t, 2., .02, population=0, kind='V')
        self.assertFalse(r['baseline_scale_qualified'])
        self.assertIsNone(scale)
        self.assertIsNone(r['peak_in_baseline_sd'])
        self.assertEqual(r['recovery_status'], 'no_detectable_response')
        self.assertEqual(r['finite_response_energy'], 0.)

    def test_truncated_recovery_does_not_count_as_sustained(self):
        self.assertIsNone(p.sustained_crossing([False] * 9 + [True], 0, 2))
        self.assertEqual(p.sustained_crossing([False, True, True], 0, 2), 1)

    def test_identical_population_response_is_one_mode(self):
        t = np.arange(5000) * .002
        delta = np.repeat(np.where(t >= 2., np.exp(-(t - 2.)), 0.)[:, None], 9, axis=1)
        r, z = p.system_response(delta, [1.] * 9, t, 2., .02, 'V')
        self.assertAlmostEqual(r['response_mode_participation_ratio'], 1.)
        self.assertAlmostEqual(r['other_population_energy_fraction'], 8. / 9.)
        self.assertEqual(z.shape[1], 9)

    def test_overflow_is_rejected_without_clipping(self):
        with self.assertRaisesRegex(ValueError, 'finite numerical range'):
            p.finite_energy(np.array([1e300, 1e300]), np.array([0., 1.]))


class PairAndContrastTests(unittest.TestCase):
    def test_actual_qualification_and_all_nine_responses(self):
        bank = bank_fixture(2)
        pop, system, proof, contrasts, disposition = p.evaluate_bank(bank, expected_pairs=2)
        self.assertEqual(len(pop), 2 * 9 * 2)
        self.assertEqual(len(system), 4)
        self.assertEqual(len(proof), 2)
        self.assertTrue(all(json.loads(r['proof_json'])['full_noise_stream_exactly_equal'] for r in proof))
        self.assertEqual(len(contrasts), 4)
        # Exact same cortical response can conceal larger non-readout responses.
        for row in contrasts:
            self.assertAlmostEqual(row['cortical_readout_relative_response_distance'], 0.)
            self.assertGreater(row['other_populations_relative_response_distance'], .5)
            self.assertGreater(row['other_minus_readout_response_distance'], .5)

    def test_preinput_hidden_population_difference_fails(self):
        bank = bank_fixture(1)
        trace = bank['traces']['candidate_000_pulse']
        trace['values'] = trace['values'].copy()
        trace['values'][100, 16] += .1
        with self.assertRaisesRegex(ValueError, 'before the input'):
            p.evaluate_bank(bank, expected_pairs=1)

    def test_different_noise_stream_fails_even_when_cortex_matches(self):
        bank = bank_fixture(1)
        bank['traces']['candidate_000_pulse']['noise'][4000] += .01
        with self.assertRaisesRegex(ValueError, 'noise streams differ'):
            p.evaluate_bank(bank, expected_pairs=1)

    def test_missing_pair_cannot_silently_change_denominator(self):
        bank = bank_fixture(2)
        bank['rows'].pop()
        with self.assertRaisesRegex(ValueError, 'Missing or duplicate'):
            p.evaluate_bank(bank, expected_pairs=2)

    def test_amplitudes_are_stratified_not_a_sweep_or_pooled_match(self):
        bank = bank_fixture(2)
        pulse = bank['traces']['candidate_001_pulse']
        pulse['row']['input_amplitude'] = 10.
        pulse['config_text'] = pulse['config_text'].replace('Amplitude: 1.0', 'Amplitude: 10.0')
        pulse['config']['input_amplitude'] = 10.
        pulse['stimulus'] *= 10.
        pop, system, proof, contrast, disposition = p.evaluate_bank(bank, expected_pairs=2)
        self.assertEqual(contrast, [])
        self.assertEqual(disposition['score:different_input'], 1)
        report = p.summarize(pop, system, proof, contrast, disposition, {'status': 'not_supplied'})
        self.assertEqual({r['input_amplitude'] for r in report['family_amplitude_strata']}, {1., 10.})
        self.assertFalse(report['inference']['linear_impulse_response_or_controllability_established'])

    def test_features_matching_is_stricter_and_boundary_inclusive(self):
        bank = bank_fixture(2)
        records = []
        for i in range(2):
            c, s = (bank['traces'][f'candidate_{i:03d}_{tag}'] for tag in ('control', 'pulse'))
            records.append(p._metadata(c, s) | {'control_row': c['row'], 'noise_stream_sha256': 'identical_test_stream'})
        records[1]['control_row']['feature_spectral_centroid_hz'] = 11.
        self.assertTrue(p.match_controls(*records, spectra=True)[0])
        records[1]['control_row']['feature_spectral_centroid_hz'] = 11.0001
        self.assertTrue(p.match_controls(*records, spectra=False)[0])
        self.assertFalse(p.match_controls(*records, spectra=True)[0])

    def test_crossconfiguration_background_noise_must_match(self):
        bank = bank_fixture(2)
        # Each pair still shares its complete stream, but the two configurations
        # have different backgrounds and cannot furnish a diagnostic contrast.
        for tag in ('control', 'pulse'):
            bank['traces']['candidate_001_' + tag]['noise'] += .1
        _, _, _, contrasts, dispositions = p.evaluate_bank(bank, expected_pairs=2)
        self.assertEqual(contrasts, [])
        self.assertEqual(dispositions['score:different_or_unverified_background_noise_stream'], 1)

    def test_clipped_score_matches_are_reported_separately(self):
        a, b = {'score': 1., 'status': 'ok'}, {'score': 1., 'status': 'ok'}
        self.assertEqual(p.clipping_stratum(a, b), 'both_ceiling')
        b['score'] = .99
        self.assertEqual(p.clipping_stratum(a, b), 'mixed_clipping')
        a['score'] = .98
        self.assertEqual(p.clipping_stratum(a, b), 'neither_clipped')

    def test_compatibility_rejects_human_test_selection(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            path = Path(tmp) / 'compatibility.json'
            path.write_text(json.dumps({'selection_uses_test_targets': True, 'selection_unit': 'training/validation',
                                        'compatible_simulation_ids': ['candidate_000_control']}))
            with self.assertRaisesRegex(ValueError, 'exclusively'):
                p.read_compatibility({'path': str(path)})
            path.write_text(json.dumps({'selection_uses_test_targets': False, 'selection_unit': 'training/validation',
                                        'compatible_simulation_ids': ['candidate_000_control']}))
            ids, receipt = p.read_compatibility({'path': str(path)})
            self.assertEqual(ids, {'candidate_000_control'})
            self.assertTrue(receipt['not_physiological_equivalence'])

    def test_shared_configuration_contrasts_never_become_human_confidence_intervals(self):
        pop, system, proof, contrasts, dispositions = p.evaluate_bank(bank_fixture(3), expected_pairs=3)
        report = p.summarize(pop, system, proof, contrasts, dispositions, {'status': 'not_supplied'})
        self.assertEqual(report['n_qualified_control_pulse_pairs'], 3)
        self.assertIn('no human sample', report['independent_units'])
        self.assertIn('no pseudo-replicated', report['inference']['uncertainty'])
        self.assertEqual(len(report['matched_response_summaries']), 4)
        self.assertTrue(all(r['n_dependent_configuration_contrasts'] == 3 for r in report['matched_response_summaries']))


if __name__ == '__main__':
    unittest.main()

