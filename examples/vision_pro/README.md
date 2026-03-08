# Vision Pro Integration Bundle

This folder contains the in-repo assets needed to hook the NVIDIA Vision Pro sample client into `BlueprintValidation`.

Files:

- `BlueprintVisionProRelayClient.swift`
  - drop-in Swift helper that opens a TCP connection to the relay and sends JSON-lines packets
- `sample_client_patch_template.diff`
  - patch template showing where and how to call the helper from a typical per-frame tracking update path
- `relay_settings.example.json`
  - example host/port values to keep the sample client and GPU-box relay aligned

This bundle does **not** include NVIDIA's sample client source code. It is meant to be copied into that repo and adjusted to the actual file/class names present there.
