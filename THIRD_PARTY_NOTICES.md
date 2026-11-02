# Third-party components

Phase 2 loads the official Microsoft WavLM Base+ checkpoint, revision
`4c66d4806a428f2e922ccfa1a962776e232d487b`, from
[Microsoft's model repository](https://huggingface.co/microsoft/wavlm-base-plus).
Its model card links to the [Microsoft unilm MIT license](https://github.com/microsoft/unilm/blob/master/LICENSE).
The weights are downloaded to an ignored local cache and are not redistributed here.
Transformers is installed as a dependency; no model repository custom code is executed.
Our frozen statistics-pooling regression head is an implementation for this project,
not a claim to reproduce MultiPA, E2E-R, or the Interspeech 2022 SSL scoring architecture.

Phase 2 downloads the official SpeechOcean762 **training partition** WAVs at the
same pinned corpus revision below. It preserves the waveform, applies the official
WavLM per-utterance normalization, and derives temporal mean/standard-deviation
features from Transformer layers. Speaker membership follows the preserved
Baseline 01 split. Corpus attribution and CC BY 4.0 terms apply to these derived
features; original waveforms and features remain in ignored local directories.

`speakingt/vendor/gopt.py` is copied without algorithm changes from
[YuanGongND/gopt](https://github.com/YuanGongND/gopt), commit
`bed909daf8eca035095871e51642525acc5b9b55`, `src/models/gopt.py`.
Copyright (c) 2022 Yuan Gong. BSD-3-Clause; full terms are included in
`speakingt/vendor/GOPT-LICENSE`. The upstream file also credits the timm/T2T-ViT
attention implementation. Attribution comments are retained.

Gong, Chen, Chu, Chang, and Glass (2022), *Transformer-Based Multi-Aspect
Multi-Granularity Non-native English Speaker Pronunciation Assessment*,
ICASSP, DOI: 10.1109/ICASSP43922.2022.9746743.

SpeechOcean762 is attributed to Zhang et al. (2021), *speechocean762: An
Open-Source Non-native English Speech Corpus For Pronunciation Assessment*,
Interspeech. [Official corpus](https://github.com/jimbozhang/speechocean762),
[OpenSLR description](https://www.openslr.org/101/),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
GOPT-format features are downloaded from the public preservation at
[JazminVidal/gopt-w](https://github.com/JazminVidal/gopt-w), pinned to
`908bc7b8258c58aa1c8193689243e6b876ff4a71`. This is a third-party source,
not an official GOPT-author download. Every human label and utterance/phone
identity is independently checked against the pinned original corpus.
Changes: restore official test membership from two stored partitions, sort
test IDs, convert floats to float32, derive an independent validation split
from official train speakers. The upstream code, data, and checkpoint each
retain their own provenance; no new license is asserted for third-party data.

No corpus recordings, feature arrays, personal recordings, transcripts, or
listener notes are redistributed in this repository. Downloaded artifacts
are cached under ignored local folders. The official GOPT LibriSpeech
checkpoint is fetched from its original repository, never silently replaced
by a locally trained model. File URLs and SHA256 digests are in
`experiments/gopt/assets.json`.
