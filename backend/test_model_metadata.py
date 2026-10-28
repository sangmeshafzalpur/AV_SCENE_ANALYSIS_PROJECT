import os
import sys
from pathlib import Path
import unittest
import requests
import tensorflow as tf

BACKEND_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.model_metadata_service import (
    build_model_metadata,
    check_model_readiness,
    compute_sha256,
    MODEL_PATH,
)
from services.model_inference_service import load_multimodal_production_model

class TestModelMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = build_model_metadata()
        cls.model = load_multimodal_production_model()

    def test_model_file_integrity(self):
        self.assertTrue(MODEL_PATH.exists(), f"Model file not found at {MODEL_PATH}")
        expected_size = 65410165
        expected_sha256 = "404f374484b2304e44074dd44a81a8130add1b22c22dcd7a7e21ea56f8763fde"
        actual_size = MODEL_PATH.stat().st_size
        actual_sha256 = compute_sha256(MODEL_PATH)
        
        self.assertEqual(actual_size, expected_size, "Model file size does not match verified size")
        self.assertEqual(actual_sha256, expected_sha256, "Model SHA-256 does not match verified checksum")
        self.assertEqual(self.metadata["model_identification"]["file_size_bytes"], expected_size)
        self.assertEqual(self.metadata["model_identification"]["sha256_checksum"], expected_sha256)

    def test_model_metadata_vs_loaded_model(self):
        # 1. Model Name
        self.assertEqual(
            self.model.name,
            self.metadata["model_identification"]["canonical_name"],
            "Model name mismatch"
        )
        
        # 2. Parameter Counts
        total_params = self.model.count_params()
        trainable_params = sum(tf.keras.backend.count_params(w) for w in self.model.trainable_weights)
        non_trainable_params = total_params - trainable_params
        
        self.assertEqual(total_params, 6455846)
        self.assertEqual(trainable_params, 2406275)
        self.assertEqual(non_trainable_params, 4049571)
        
        param_meta = self.metadata["verified_facts"]["parameters"]
        self.assertEqual(param_meta["total_parameters"], total_params)
        self.assertEqual(param_meta["trainable_parameters"], trainable_params)
        self.assertEqual(param_meta["non_trainable_parameters"], non_trainable_params)

        # 3. Input Signatures
        self.assertEqual(len(self.model.inputs), 4)
        input_meta = self.metadata["verified_facts"]["input_tensors"]
        self.assertEqual(len(input_meta), 4)
        
        expected_inputs = [
            ("video", [None, 12, 224, 224, 3]),
            ("audio", [None, 64, 313, 1]),
            ("audio_mask", [None, 1]),
            ("text", [None]),
        ]
        
        for i, (exp_name, exp_shape) in enumerate(expected_inputs):
            actual_inp = self.model.inputs[i]
            meta_inp = input_meta[i]
            self.assertEqual(actual_inp.name, exp_name)
            self.assertEqual(meta_inp["name"], exp_name)
            self.assertEqual(list(actual_inp.shape), exp_shape)
            self.assertEqual(meta_inp["shape"], exp_shape)

        # 4. Output Signature
        self.assertEqual(len(self.model.outputs), 1)
        output_meta = self.metadata["verified_facts"]["output_tensor"]
        self.assertEqual(list(self.model.outputs[0].shape), [None, 3])
        self.assertEqual(output_meta["shape"], [None, 3])
        self.assertEqual(output_meta["activation"], "softmax")

        # 5. Layer Count
        self.assertEqual(len(self.model.layers), 26)
        self.assertEqual(self.metadata["verified_facts"]["architecture_layers"]["total_layers"], 26)

    def test_class_mapping_consistency(self):
        class_mapping = self.metadata["verified_facts"]["class_mapping"]
        self.assertEqual(class_mapping["0"], "Accident")
        self.assertEqual(class_mapping["1"], "Fire")
        self.assertEqual(class_mapping["2"], "Normal")

    def test_metadata_categorical_separation(self):
        # Strict validation that facts, configs, measured results, and unknowns are separated
        self.assertIn("verified_facts", self.metadata)
        self.assertIn("configured_values", self.metadata)
        self.assertIn("measured_results", self.metadata)
        self.assertIn("unknown_or_untested_information", self.metadata)
        self.assertIn("limitations", self.metadata)

        # Verify ground truth test metrics in measured_results
        measured = self.metadata["measured_results"]["blind_test_evaluation"]
        self.assertEqual(measured["total_test_videos"], 270)
        self.assertEqual(measured["correct_predictions"], 265)
        self.assertEqual(measured["incorrect_predictions"], 5)
        self.assertAlmostEqual(measured["accuracy"], 0.981481, places=4)

        # Verify unknowns are explicitly declared
        unknowns = self.metadata["unknown_or_untested_information"]
        self.assertIn("english_audio_accuracy", unknowns)
        self.assertIn("modality_ablation_baselines", unknowns)

    def test_api_endpoints_live(self):
        base_url = "http://127.0.0.1:8000"
        try:
            r_check = requests.get(f"{base_url}/api/health", timeout=1)
            if r_check.status_code != 200:
                raise requests.RequestException()
        except Exception:
            raise unittest.SkipTest("Live API server is not running on 127.0.0.1:8000 (skipping live test)")
        
        # 1. Metadata endpoint
        r_meta = requests.get(f"{base_url}/api/model/metadata", timeout=5)
        self.assertEqual(r_meta.status_code, 200)
        meta_json = r_meta.json()
        self.assertEqual(meta_json["model_identification"]["canonical_name"], "EfficientNetB0_BiLSTM_AudioCNN_KannadaASR_TextBiLSTM")
        self.assertEqual(meta_json["verified_facts"]["parameters"]["total_parameters"], 6455846)

        # 2. Readiness endpoint
        r_ready = requests.get(f"{base_url}/api/model/readiness", timeout=5)
        self.assertEqual(r_ready.status_code, 200)
        ready_json = r_ready.json()
        self.assertEqual(ready_json["status"], "ready")
        self.assertTrue(ready_json["ready"])
        for comp_name, comp_info in ready_json["components"].items():
            expected_status = "ready" if comp_name == "model_file" else "loaded"
            self.assertEqual(comp_info["status"], expected_status, f"Component {comp_name} status was {comp_info['status']}")

if __name__ == "__main__":
    unittest.main()
