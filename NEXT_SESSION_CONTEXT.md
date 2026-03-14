# Next Session Context

## Primary Goal
Turn the public hosted flow on `tryblueprint.io` for `siteworld-f5fd54898cfb` into a truly interactive world-model experience.

Current high-level target:
- replace the current preview renderer with a true pose-driven novel-view renderer
- use the full video + ARKit capture/reconstruction pipeline as the rendering basis
- keep the current image/generative path as optional refinement, not the primary source of viewpoint changes

## Repos Involved
- Runtime/backend: `/Users/nijelhunt_1/workspace/BlueprintValidation`
- Public web app: `/Users/nijelhunt_1/workspace/Blueprint-WebApp`

## What Was Completed

### BlueprintValidation
Pushed to `main`:
- `4c0856c` `Bound runtime render latency and flatten action traces`
- `3500e21` `Add immediate preview renders with async refinement`

Implemented:
- bounded runtime render timeouts
- granular runtime logs around reset/render/refinement
- Firestore-safe `actionTrace` payloads
- immediate preview mode for reset/step
- async render refinement in the background
- refinement coalescing to latest snapshot

Runtime behavior now:
- `reset` and `step` return fast preview frames instead of waiting on the 45s render timeout path
- `qualityFlags.render_mode = "preview"`
- `qualityFlags.refinement_status = "pending"` on fast responses
- refinement runs asynchronously

### Blueprint-WebApp
Pushed to `main`:
- `9eff2e1` `Smooth hosted session startup and mirroring`
- `67f1c6e` `Prefer runtime state for live hosted session reads`
- `650fd9f` `Add Redis-backed hosted session live store`
- `d40bdcb` `Document Redis live-session configuration`
- `67f303f` `Expose hosted session live-store backend in health`

Implemented:
- removed the initial public workspace race where `/render` ran before `/reset`
- moved Firestore out of the synchronous `reset`/`step` hot path
- active session reads now prefer runtime `/state`
- Redis-backed shared live session store when `REDIS_URL` is configured
- memory fallback when Redis is not configured
- health/debug signal exposing the live session backend

## Production Verification Already Completed

### Web App Health
Verified live:
- `https://tryblueprint.io/health/status`
- `https://tryblueprint.io/health/ready`

Observed:
- `liveSessionStore.backend = "redis"`
- `redisConfigured = true`
- `redisConnected = true`

### Fresh Public Session
Verified on `tryblueprint.io`:
- start URL: `https://tryblueprint.io/site-worlds/siteworld-f5fd54898cfb/start`

Fresh verified session:
- `c07e2bc7-802f-4ef5-a81d-90bc02a61f92`

Verified flow:
- create: success
- reset: success
- render: `200` PNG
- step: `200`
- render again: `200` PNG

Observed public step latency:
- about `2635ms` from browser fetch timing

### Runtime Host
Runtime host:
- host: `146.115.17.157`
- ssh port: `45408`
- user: `root`
- runtime public base: `http://146.115.17.157:45457`
- runtime repo on host: `/workspace/blueprint-neoverse-run/BlueprintValidation`

Observed in runtime logs:
- public web app repeatedly hits runtime `/state`
- `reset_session.preview_ready`
- `step_session.preview_ready`
- `render_refinement.start`

## Current Architecture State

### What Is Good Now
- hosted flow no longer hard-hangs
- live session reads do not depend on Firestore freshness
- Firestore is no longer on the synchronous hot path
- Redis is active in production for live-session state
- preview-first interaction is materially faster

### What Is Still Not Good Enough
- the displayed image often does not visibly change when clicking `Move forward`, `Turn left`, `Reach in`
- backend state changes, but visual feedback is weak
- the current preview path is still based on the base/canonical frame, not true pose-driven novel view synthesis
- refinement is still image/generator driven, not a true world-model renderer
- websocket/SSE push is still not implemented

## Important Technical Reason For The Current UX Limitation
The current fast preview path in `BlueprintValidation` is not a real viewpoint-changing renderer.

The preview path:
- uses the stored base/canonical frame as the visual source
- composites on top of that frame
- returns quickly, but does not produce strong viewpoint changes from action updates

The current refinement path:
- still goes through the NeoVerse wrapper flow
- uses a conditioning input and trajectory labels
- is not a true pose-driven renderer from a persistent 3D scene representation

This is why step count and reward change while the visible image often barely changes.

## Next Major Task
Replace the preview renderer with a true pose-driven novel-view renderer based on the full video + ARKit reconstruction pipeline.

Desired direction:
- use the full capture, not a single frame, as the rendering basis
- use ARKit poses/intrinsics/depth/scene data directly
- update camera/robot/world pose on every step
- render a new view immediately from that spatial representation
- keep the current generative/image model path only as optional refinement

## Likely Files To Start With

### BlueprintValidation
- `/Users/nijelhunt_1/workspace/BlueprintValidation/src/blueprint_validation/neoverse_production_runtime.py`
- `/Users/nijelhunt_1/workspace/BlueprintValidation/src/blueprint_validation/neoverse_runner_wrapper.py`
- `/Users/nijelhunt_1/workspace/BlueprintValidation/src/blueprint_validation/runtime_layer_grounding.py`

### Blueprint-WebApp
- `/Users/nijelhunt_1/workspace/Blueprint-WebApp/client/src/pages/HostedSessionWorkspace.tsx`
- `/Users/nijelhunt_1/workspace/Blueprint-WebApp/server/routes/site-world-sessions.ts`

## Suggested Next-Session Prompt

```text
Work in:
- /Users/nijelhunt_1/workspace/BlueprintValidation
- /Users/nijelhunt_1/workspace/Blueprint-WebApp

Context:
- The hosted public flow on tryblueprint.io for siteworld-f5fd54898cfb now works end-to-end.
- Runtime reset/step are fast because we implemented immediate preview + async refinement.
- Firestore is off the synchronous hot path.
- Active hosted session reads prefer runtime state and Redis-backed live session state.
- Production health now reports liveSessionStore.backend=redis.

Verified production state:
- https://tryblueprint.io/health/status reports redis active.
- Fresh public session create -> reset -> render -> step -> render succeeds.
- Runtime logs show /state reads from the web app and preview_ready/refinement activity.

Remaining problem:
- The image shown in the live runtime viewport does not meaningfully change when the user clicks Move forward / Turn left / Reach in.
- Backend state changes, but the visual renderer is still effectively base-frame driven.
- This is not yet a true interactive world model.

What to do next:
1. Replace the current preview renderer with a true pose-driven novel-view renderer.
2. Base the renderer on the full video + ARKit reconstruction pipeline, not on a single frame.
3. Make each step action visibly change the displayed viewpoint.
4. Keep the existing generator/image-model path only as optional refinement.
5. Preserve the current fast UX: immediate response first, refinement second.

Success criteria:
- Clicking Move forward / Turn left / Reach in causes immediate visible changes in the viewport.
- Viewpoint changes are driven by pose updates, not just metadata changes.
- The renderer uses the whole capture + ARKit data path as the primary scene basis.
- Public hosted flow remains fast and stable.
```
