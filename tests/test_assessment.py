"""Transport-contract tests only: no uploads and no claims of model accuracy."""

import base64
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import wave

from speakingt.assessment import AssessmentError, assess_azure


class AssessmentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "input.wav"
        self.write_wav()
        self.environment = patch.dict(os.environ, {
            "SPEAKINGT_AZURE_KEY": "not-a-real-key",
            "SPEAKINGT_AZURE_REGION": "eastus",
        })
        self.environment.start()
        self.transport = patch("speakingt.assessment.http.client.HTTPSConnection")
        self.constructor = self.transport.start()
        self.connection = self.constructor.return_value
        self.response = self.connection.getresponse.return_value
        self.response.status = 200
        self.result = {
            "RecognitionStatus": "Success",
            "NBest": [{
                "Display": "Good morning.",
                "AccuracyScore": 90.0,
                "FluencyScore": 85.0,
                "CompletenessScore": 100.0,
                "PronScore": 88.0,
                "ProsodyScore": 80.0,
                "Words": [{
                    "Word": "good", "Offset": 1000000, "Duration": 3000000,
                    "AccuracyScore": 92.0, "ErrorType": "None",
                    "Phonemes": [{"Phoneme": "g", "Offset": 1000000,
                                  "Duration": 800000, "AccuracyScore": 95.0}],
                }],
            }],
        }
        self.set_response(self.result)

    def tearDown(self):
        self.transport.stop()
        self.environment.stop()
        self.directory.cleanup()

    def write_wav(self, seconds=1, rate=16000, width=2, channels=1):
        with wave.open(str(self.path), "wb") as target:
            target.setnchannels(channels)
            target.setsampwidth(width)
            target.setframerate(rate)
            target.writeframes(b"\0" * (int(seconds * rate) * width * channels))

    def set_response(self, value):
        self.response.read.return_value = json.dumps(value).encode("utf-8")

    def test_request_contract_and_normalized_phoneme_scores(self):
        result = assess_azure(self.path, "Good morning.")
        self.constructor.assert_called_once_with("eastus.stt.speech.microsoft.com", timeout=45)
        args, kwargs = self.connection.request.call_args
        self.assertEqual(args[0], "POST")
        self.assertIn("language=en-US&format=detailed", args[1])
        headers = kwargs["headers"]
        self.assertEqual(headers["Ocp-Apim-Subscription-Key"], "not-a-real-key")
        self.assertEqual(headers["Content-Type"], "audio/wav; codecs=audio/pcm; samplerate=16000")
        self.assertEqual(headers["Accept"], "application/json")
        parameters = json.loads(base64.b64decode(headers["Pronunciation-Assessment"]))
        self.assertEqual(parameters["ReferenceText"], "Good morning.")
        self.assertEqual(parameters["Granularity"], "Phoneme")
        self.assertEqual(parameters["GradingSystem"], "HundredMark")
        self.assertTrue(parameters["EnableMiscue"])
        self.assertTrue(parameters["EnableProsodyAssessment"])
        with wave.open(io.BytesIO(kwargs["body"]), "rb") as sent:
            self.assertEqual((sent.getnchannels(), sent.getsampwidth(), sent.getframerate()), (1, 2, 16000))
            self.assertEqual(sent.getnframes(), 16000)
        self.assertEqual(result["provider"], "azure_pronunciation_assessment")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["scores"]["accuracy"], 90)
        word = result["words"][0]
        self.assertEqual((word["start_s"], word["end_s"]), (0.1, 0.4))
        self.assertEqual(word["phonemes"][0]["accuracy"], 95)
        self.assertEqual(word["phonemes"][0]["end_s"], 0.18)
        self.connection.close.assert_called_once()
        json.dumps(result, allow_nan=False)

    def test_no_match_is_missing_assessment_not_zero_score(self):
        self.set_response({"RecognitionStatus": "NoMatch"})
        result = assess_azure(self.path, "Hello.")
        self.assertEqual(result["status"], "no_match")
        self.assertEqual(result["scores"], {})
        self.assertEqual(result["words"], [])

    def test_unsupported_audio_is_rejected_before_upload(self):
        for options in ({"seconds": 30.01}, {"seconds": 0}, {"rate": 44100}, {"width": 1}, {"channels": 2}):
            with self.subTest(options=options):
                self.write_wav(**options)
                with self.assertRaises(AssessmentError):
                    assess_azure(self.path, "Hello.")
        self.constructor.assert_not_called()

    def test_truncated_wav_is_rejected(self):
        self.path.write_bytes(self.path.read_bytes()[:-20])
        with self.assertRaisesRegex(AssessmentError, "truncated"):
            assess_azure(self.path, "Hello.")
        self.constructor.assert_not_called()

    def test_bad_configuration_cannot_change_host_or_add_headers(self):
        for region in ("https://evil.example", "eastus.evil.example", "eastus@evil", "eastus/evil", "eastus\r\nX: x", ""):
            with self.subTest(region=region), patch.dict(os.environ, {"SPEAKINGT_AZURE_REGION": region}):
                with self.assertRaises(AssessmentError):
                    assess_azure(self.path, "Hello.")
        with patch.dict(os.environ, {"SPEAKINGT_AZURE_KEY": "bad\r\nX: value"}):
            with self.assertRaises(AssessmentError):
                assess_azure(self.path, "Hello.")
        self.constructor.assert_not_called()

    def test_reference_required_before_network(self):
        for reference in ("", "   ", "x" * 4001):
            with self.subTest(reference_length=len(reference)), self.assertRaises(AssessmentError):
                assess_azure(self.path, reference)
        self.constructor.assert_not_called()

    def test_http_errors_do_not_echo_response_or_retry_redirect(self):
        self.response.read.return_value = b"not-a-real-key private provider message"
        for status in (301, 302, 307, 308, 401, 403, 429, 500):
            with self.subTest(status=status):
                self.response.status = status
                self.connection.request.reset_mock()
                with self.assertRaises(AssessmentError) as raised:
                    assess_azure(self.path, "Hello.")
                self.assertNotIn("not-a-real-key", str(raised.exception))
                self.connection.request.assert_called_once()
        self.response.read.assert_not_called()

    def test_connection_error_is_sanitized(self):
        self.connection.request.side_effect = OSError("not-a-real-key")
        with self.assertRaisesRegex(AssessmentError, "connection failed") as raised:
            assess_azure(self.path, "Hello.")
        self.assertNotIn("not-a-real-key", str(raised.exception))
        self.connection.close.assert_called_once()

    def test_invalid_provider_payload_is_not_accepted(self):
        for payload in ([], {"RecognitionStatus": "Success", "NBest": []},
                        {"RecognitionStatus": "Success", "NBest": [{"Display": "Only ASR"}]}):
            with self.subTest(payload=payload):
                self.set_response(payload)
                with self.assertRaises(AssessmentError):
                    assess_azure(self.path, "Hello.")
        self.result["NBest"][0]["AccuracyScore"] = float("nan")
        self.set_response(self.result)
        with self.assertRaises(AssessmentError):
            assess_azure(self.path, "Hello.")

    def test_invalid_json_is_sanitized(self):
        self.response.read.return_value = b"not-a-real-key"
        with self.assertRaisesRegex(AssessmentError, "invalid JSON") as raised:
            assess_azure(self.path, "Hello.")
        self.assertNotIn("not-a-real-key", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
