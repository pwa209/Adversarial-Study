"""Protect exclusive audit slots from invalid scalar/NaN proof serialization."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import audit_ops as ops
import native_probe


class ReceiptSerializationTests(unittest.TestCase):
    def test_numpy_scalar_boundary_keeps_booleans_and_numbers_typed(self):
        raw={'proof':[np.bool_(True),np.float64(.2),np.int64(42)],'nested':{'pass':np.bool_(False)}}
        proof=json.loads(json.dumps(raw,default=native_probe.numpy_scalar,allow_nan=False))
        self.assertEqual(proof,{'proof':[True,.2,42],'nested':{'pass':False}})
        self.assertIs(type(proof['proof'][0]),bool)

    def test_probe_rejects_unknown_objects_and_nonfinite_values(self):
        for value in (object(),np.asarray([1.,2.])):
            with self.assertRaises(TypeError):json.dumps(value,default=native_probe.numpy_scalar,allow_nan=False)
        with self.assertRaises(ValueError):json.dumps({'p':np.float64(np.nan)},default=native_probe.numpy_scalar,allow_nan=False)

    def test_invalid_payload_cannot_touch_a_receipt_slot(self):
        with patch.object(ops,'state') as state:
            for payload in ({'unexpected':np.bool_(True)},{'invalid':float('nan')}):
                with self.assertRaises((ValueError,TypeError)):ops.write_slot(Path('never_open.json'),payload)
            state.assert_not_called()

    def test_valid_payload_written_once_to_actual_original_inode(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'slot.json';path.touch();before=path.stat()
            state={'identities':{str(path):[before.st_dev,before.st_ino]}}
            with patch.object(ops,'state',return_value=state),patch.object(ops,'UID',before.st_uid):
                ops.write_slot(path,{'passed':True,'value':42})
                self.assertEqual(json.loads(path.read_text()),{'passed':True,'value':42})
                with self.assertRaisesRegex(ValueError,'already written'):ops.write_slot(path,{'overwritten':True})
            self.assertEqual([path.stat().st_dev,path.stat().st_ino],[before.st_dev,before.st_ino])


if __name__=='__main__':unittest.main()
