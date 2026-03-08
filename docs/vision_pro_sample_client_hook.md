# Vision Pro Sample Client Hook

This document defines the exact client-side hook for the NVIDIA Vision Pro sample client.

The goal is simple:

- capture hand or controller deltas in the sample client
- serialize them into a small JSON packet
- send them over TCP to `run-vision-pro-relay`

The relay already exists in this repo and normalizes those packets into the action format used by `record-teleop --teleop-device vision_pro`.

## What You Add To The Sample Client

Add the Swift helper from:

- [examples/vision_pro/BlueprintVisionProRelayClient.swift](/Users/nijelhunt_1/workspace/BlueprintValidation/examples/vision_pro/BlueprintVisionProRelayClient.swift)

And use the concrete patch template from:

- [examples/vision_pro/sample_client_patch_template.diff](/Users/nijelhunt_1/workspace/BlueprintValidation/examples/vision_pro/sample_client_patch_template.diff)

Reference relay settings:

- [examples/vision_pro/relay_settings.example.json](/Users/nijelhunt_1/workspace/BlueprintValidation/examples/vision_pro/relay_settings.example.json)

This gives you:

- `BlueprintVisionProRelayClient`
- `BlueprintTeleopPoseMapper`

## Where To Capture Deltas

Use the hook point where the sample client already has fresh hand/controller tracking data each frame.

For the official NVIDIA path, that means the place in the Vision Pro sample client where:

- hand tracking is updated
- and/or controller pose updates are processed

Per NVIDIA's docs, the Vision Pro sample client already supports:

- hand tracking
- CloudXR streaming from the remote runtime

So the correct hook is:

1. **after** you receive the newest hand/controller tracking pose for the current frame
2. **before** or alongside any existing local visualization / debug handling
3. **every frame** while teleop is active

In plain terms:

- if the sample client has a callback like `onHandTrackingUpdated(...)`, hook there
- if it exposes a per-frame update loop with current tracked pose state, hook there

Do **not** hook after gesture recognition only. You want the continuous pose stream, not just discrete button-like events.

## What To Send

Preferred packet shape from the sample client:

```json
{
  "right_hand": {
    "translation": [dx, dy, dz],
    "rotation_rpy": [droll, dpitch, dyaw]
  },
  "gestures": {
    "pinch": true
  },
  "done": false
}
```

This is the best fit because:

- the repo-side relay already understands it
- it matches the Franka `ee_delta_pose + gripper` control contract cleanly

If you already have direct EE deltas instead of hand deltas, you can send:

```json
{
  "ee_delta_pose": [dx, dy, dz, droll, dpitch, dyaw],
  "gripper_delta": -1.0,
  "done": false
}
```

To end recording:

```json
{"done": true}
```

## How To Serialize

Use the provided Swift `Codable` structs:

- `BlueprintHandPosePacket`
- `BlueprintPosePacket`
- `BlueprintDirectActionPacket`

`BlueprintVisionProRelayClient` handles:

- JSON encoding
- appending newline delimiters
- sending over TCP

## How To Send Over TCP JSON-lines

On the Vision Pro side:

```swift
let relay = BlueprintVisionProRelayClient(host: "GPU_BOX_IP", port: 49111)
relay.start()
```

Each frame:

```swift
relay.sendHandPose(
    translation: SIMD3<Float>(dx, dy, dz),
    rotationRPY: SIMD3<Float>(droll, dpitch, dyaw),
    pinch: pinchActive
)
```

When teleop ends:

```swift
relay.finish()
relay.stop()
```

## Exact Integration Pattern

The most practical pattern is:

1. Create one `BlueprintVisionProRelayClient` when teleop mode starts
2. Create one `BlueprintTeleopPoseMapper`
3. For every tracked frame:
   - read current right-hand pose
   - convert pose to translation + Euler rotation
   - call `deltaPacket(...)` on the mapper
   - send that packet through the relay client
4. On session end, call `finish()`

Example:

```swift
var mapper = BlueprintTeleopPoseMapper()
let relay = BlueprintVisionProRelayClient(host: "GPU_BOX_IP", port: 49111)
relay.start()

func onHandTrackingFrame(position: SIMD3<Float>, rotationRPY: SIMD3<Float>, pinch: Bool) {
    let packet = mapper.deltaPacket(
        currentTranslation: position,
        currentRotationRPY: rotationRPY,
        pinch: pinch
    )

    relay.sendHandPose(
        translation: SIMD3<Float>(
            packet.right_hand.translation[0],
            packet.right_hand.translation[1],
            packet.right_hand.translation[2]
        ),
        rotationRPY: SIMD3<Float>(
            packet.right_hand.rotation_rpy[0],
            packet.right_hand.rotation_rpy[1],
            packet.right_hand.rotation_rpy[2]
        ),
        pinch: packet.gestures.pinch
    )
}
```

## Remote Side Expectations

On the GPU box, run both:

- `bash scripts/run_isaac_record_teleop.sh`
- `bash scripts/run_vision_pro_relay.sh`

The relay listens on:

- `BRIDGE_HOST:BRIDGE_PORT` default `0.0.0.0:49111`

and forwards into the recorder at:

- `TARGET_HOST:TARGET_PORT` default `127.0.0.1:49110`

## Important Note

This is the exact hook and packet contract for integrating with the sample client.

What is still outside this repo:

- the actual Vision Pro sample client source tree
- the exact file/class names inside NVIDIA's sample app

So the work left on the Vision Pro side is mechanical:

- add the Swift file
- call it from the frame update where tracked pose data already exists
- point it at the GPU box IP and relay port
