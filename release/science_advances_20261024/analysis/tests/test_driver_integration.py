"""Check completed-parent adoption, immutable inputs and bounded sibling execution."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import driver


class CompletedParentTests(unittest.TestCase):
    def fixture(self):
        files={name:{'sha256':name, 'path':'/existing/native/receipt.json'} for name in
               ('native_cache_receipt','bridge_compatibility','bank_receipt','bank_inventory','scored_bank')}
        value={'files':files,'native_cache_dir':'/existing/native'}
        parent={'status':'full_population_system_completed','job_id':'6963704',
                'stages':{name:{'returncode':0} for name in ('native_population','population_bridge','population_perturbations')},
                'outputs':{'native_population/receipt.json':'native_cache_receipt','population_bridge/compatibility.json':'bridge_compatibility'}}
        cache={'schema':'native_nine_population_cache_v2','status':'native_population_cache_qualified',
               'inventoried_rows':664,'qualified_rows':664,'bank_receipt_sha256':'bank_receipt',
               'inventory_sha256':'bank_inventory','scored_bank_sha256':'scored_bank',
               'output_time_labels_preserved':True,'output_clock_rebased':False}
        return value,parent,cache

    def test_completed_cache_adopted_without_regeneration(self):
        proof=driver.parent_cache_gate(*self.fixture())
        self.assertFalse(proof['native_cache_regenerated'])
        self.assertEqual(proof['completed_population_job_adopted'],'6963704')

    def test_incomplete_or_substituted_cache_cannot_be_adopted(self):
        for field,replacement in (('qualified_rows',663),('output_clock_rebased',True),
                                  ('bank_receipt_sha256','different'),('schema','old')):
            value,parent,cache=self.fixture();cache[field]=replacement
            with self.subTest(field=field),self.assertRaises(ValueError):
                driver.parent_cache_gate(value,parent,cache)

    def test_completed_output_hashes_are_mandatory(self):
        value,parent,cache=self.fixture();parent['outputs']['population_bridge/compatibility.json']='different'
        with self.assertRaises(ValueError):driver.parent_cache_gate(value,parent,cache)

    def test_failed_parent_stage_is_rejected(self):
        value,parent,cache=self.fixture();parent['stages']['population_bridge']['returncode']=1
        with self.assertRaises(ValueError):driver.parent_cache_gate(value,parent,cache)

    def test_stage_inputs_always_pin_completed_cache_and_fixed_bootstrap(self):
        value,_,_=self.fixture()
        inputs=driver.stage_inputs('mechanism_compensation',value)
        self.assertEqual(inputs['bootstrap_repetitions'],2000)
        self.assertEqual(inputs['files']['population_cache']['receipt_sha256'],'native_cache_receipt')
        self.assertEqual(driver.stage_inputs('pathway_balance',value)['files']['bridge_compatibility']['sha256'],
                         'bridge_compatibility')


class IndependentChildrenTests(unittest.TestCase):
    def run_case(self,script,seconds=10):
        directory=tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root=Path(directory.name);(root/'driver.py').write_text(script)
        output=root/'output';output.mkdir()
        return root,output,seconds

    def test_all_children_complete_and_streams_are_preserved(self):
        root,output,_=self.run_case('print("completed fixture")\n')
        with patch.object(driver,'HERE',root):
            records=driver.execute_branches(output,time.monotonic()+40)
        self.assertEqual(set(records),set(driver.STAGES))
        self.assertTrue(all(v['returncode']==0 for v in records.values()))
        self.assertTrue(all((output/(stage+'.stdout.log')).read_text().strip()=='completed fixture' for stage in driver.STAGES))

    def test_failed_child_preserves_successful_siblings_and_actual_returncodes(self):
        script='import sys\nfrom pathlib import Path\nname=sys.argv[sys.argv.index("--branch")+1]\nPath(sys.argv[sys.argv.index("--output")+1]+".finished").write_text(name)\nsys.exit(7 if name=="human_spatial" else 0)\n'
        root,output,_=self.run_case(script)
        with patch.object(driver,'HERE',root),self.assertRaisesRegex(ValueError,'preserved'):
            driver.execute_branches(output,time.monotonic()+40)
        records=json.loads((output/'child_execution.json').read_text())
        self.assertEqual(records['human_spatial']['returncode'],7)
        self.assertEqual(records['pathway_balance']['returncode'],0)
        self.assertTrue((output/'mechanism_compensation.finished').exists())


if __name__=='__main__':unittest.main()
