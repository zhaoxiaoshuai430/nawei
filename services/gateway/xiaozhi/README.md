# Nawei robot integration

## Local entry points

- OTA discovery: `POST /xiaozhi/ota/`
- Activation polling: `POST /xiaozhi/ota/activate` (202 pending, 200 bound)
- Device transport: `WS /xiaozhi/v1/`
- Management: `POST /api/v1/devices/bind`, `GET /api/v1/devices`,
  `GET/PATCH /api/v1/devices/{device_id}`, `DELETE /api/v1/devices/{device_id}/binding`

Gateway reads process environment first, then `.env.robot`, then `.env`.
The robot overlay is ignored by Git. The current overlay enables local testing.
Use one Gateway worker: active connections and admission limits are process-local.
Device records persist in `.runtime/devices.sqlite3`, separate from LLM Memory in
PostgreSQL/Redis. Back up the SQLite database while Gateway is stopped. The
production container uses the `gateway_devices` volume at `/var/lib/nawei`.

## Provisioning and identity

The backend requires a stable per-device provisioning secret in `X-Device-Key`
on OTA requests, alongside MAC `Device-Id` and persistent UUID `Client-Id`.
The simulator generates a 32-byte secret and sends its hex encoding. A firmware
integration must generate and persist its own secret before first registration.
The server stores the key hash, never the key. MAC/UUID alone cannot retrieve
a bound device credential. This is first-use registration, not hardware attestation.

The bundled `firmware/xiaozhi-esp32/main/ota.cc` currently sends Device-Id and
Client-Id but does not generate, store, or send X-Device-Key. Changing only its
OTA URL is insufficient for the default backend: registration returns 401.
The protocol simulator passes because it implements this required header;
its success is not proof that the bundled firmware can activate unchanged.

OTA returns a six-digit activation code for an unbound device. Codes expire after
600 seconds, are consumed transactionally, and cannot overwrite an existing owner.
After binding, OTA returns `websocket.url`, `websocket.token`, and `websocket.version`.
The device token is independent from user JWTs and bound to MAC, UUID and a
per-binding random nonce. Only its hash is stored. OTA responses use `no-store`.

To bind, authenticate using the existing Gateway user scheme and post:

```json
{"code":"123456","name":"My Nawei"}
```

Local development uses `X-User-ID`. Public deployment requires the existing HS256
JWT scheme. There is no device management webpage. Use the API directly, for
example on the development machine:

```powershell
$body = @{ code = '123456'; name = 'My Nawei' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/devices/bind -Headers @{ 'X-User-ID' = '002' } -ContentType application/json -Body $body
```

Replace the example code with the robot's current activation code. Production
uses `Authorization: Bearer <user JWT>` instead of `X-User-ID`. An operator can
issue a token on the server using `deploy/scripts/create_jwt.py` with the Gateway
JWT environment, or inside the running Gateway container:

```sh
docker compose --env-file deploy/.env.production -f deploy/docker-compose.production.yml exec -T gateway python /app/deploy/scripts/create_jwt.py --user-id 002 --expires-in 900
```

The printed token is private. Bind over the server's authenticated HTTPS API.

Device updates accept `name`, `voice` (speaker ID), `role` (`companion` or `tutor`),
and `enabled`. The role is passed as a constrained voice-style preference to the
existing LLM Agent. Speaker IDs must exist in the TTS installation. Changes apply
on the next turn. Disabling/unbinding disconnects the active session immediately.
Unbinding revokes the token and permits the next device initialization to register
a fresh NVS key/UUID. Unbind in the account before clearing NVS on a bound robot;
a factory reset by itself does not remove server ownership.

## Audio protocol

The WebSocket handshake requires `Authorization: Bearer nwdev_...`,
`Protocol-Version: 1`, `Device-Id`, and `Client-Id`.
Send this within ten seconds:

```json
{"type":"hello","version":1,"transport":"websocket","audio_params":{"format":"opus","sample_rate":16000,"channels":1,"frame_duration":60},"features":{"barge_in":true,"full_duplex_transport":true}}
```

Server hello advertises 24kHz mono 60ms Opus by default. An optional
`output_audio_params` object with the same fields can explicitly select 16kHz.
The client's input `sample_rate=16000` describes uplink, not downlink capability.

Send `listen` with `state=start` and `mode=manual/auto/realtime`, then one raw Opus
packet per binary frame. In manual mode, send `listen/state=stop` to finalize ASR.
Auto/realtime modes finalize on the ASR VAD result. A client may send another
`listen/state=start` while TTS is playing; Gateway cancels the old generation,
clears queued output, and sends `tts/state=stop` with `reason=barge_in`.
Acoustic echo cancellation remains a client/firmware responsibility; MCP and
v2/v3 are not implemented.

Output order is `stt`, `tts/start`, `tts/sentence_start`, binary audio,
`tts/stop`. A single writer sends everything through a 64-item queue, paces audio
at 60ms, and discards stale generations. PyAV/FFmpeg provides Opus and resampling.
TTS returns raw PCM16; WAV containers are never forwarded as Opus. Resampling is
stateful, with one buffer/encoder per turn and a padded final frame.

Application JSON ping runs every 25 seconds, with transport ping/pong supplied by
Uvicorn (25-second ping interval and 25-second pong timeout). Idle connections
remain open while transport heartbeats succeed; older firmware need not reply
to application JSON ping. New authenticated connections replace old ones. Binary packets are
limited to 1275 bytes, JSON commands to 4096 bytes, OTA bodies to 16KiB, ASR input
to 60 seconds, and a whole turn to 180 seconds. Gateway's transport maximum is
1MiB for the generic realtime API. Limits are also applied to OTA,
binding, and new device connections. Proxy headers are trusted only from the
configured trusted proxy; do not trust arbitrary forwarded client IPs.

## Direct cloud access

Deploy the complete backend using `deploy/docker-compose.production.yml` and
`deploy/.env.production`. Set `XIAOZHI_ENABLED=true`, `NAWEI_PUBLIC_ROBOT_URL`
and `XIAOZHI_PUBLIC_WEBSOCKET_URL` to the cloud server HTTPS/WSS endpoints.
Production still requires JWT for management and WSS for device transport.
Only the ingress proxy needs a public port; model services and databases stay
on the private container network.

The existing Caddy configuration targets a DNS name. Direct public IP deployment
still needs a matching trusted IP certificate and proxy configuration. These
server-specific settings are pending; local startup does not configure them.
Run the local startup/restart scripts without extra public-profile switches.

Firmware lives in `firmware/xiaozhi-esp32`; there is no `ROBOT/` directory in
this repository. Its Kconfig OTA default is an upstream endpoint, not blank.
Before building, select the actual board and configure `CONFIG_OTA_URL` as
`https://<your-domain>/xiaozhi/ota/` and `CONFIG_NAWEI_SERVER_MODE=y` through
the board configuration/menuconfig workflow. This mode ignores `wifi/ota_url`
stored in NVS. The OTA response supplies the configured WSS address. Verify
certificate trust and implement the provisioning secret described above before
attempting physical activation.
The current firmware requires ESP-IDF >= 6.0.1; 6.1 is recommended and required
for ESP32-S31. See the firmware README and use `scripts/build.py` inside that
firmware directory; the old IDF 5.x and `ROBOT/scripts/build_nawei.ps1` instructions
do not apply to this source tree.
Production NVS encryption/flash protection needs a separate hardware decision;
this change does not burn eFuses or claim encrypted NVS.

## Verification and recovery

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_xiaozhi.py -q
.\.venv\Scripts\python.exe scripts/simulate_robot.py --rounds 20
.\.venv\Scripts\python.exe scripts/check_full_flow.py --output .runtime/full-flow-current
.\.venv\Scripts\python.exe scripts/check_full_flow.py --response-policy-only --output .runtime/response-policy
.\.venv\Scripts\python.exe scripts/simulate_robot.py --rounds 2 --mode auto --output-rate 16000
.\.venv\Scripts\python.exe scripts/check_full_flow.py --restart-only --output .runtime/full-flow-current/restart
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/status_project.ps1
```

The simulator generates speech using real TTS, encodes it to 16kHz Opus, binds a
temporary device, exercises the real services, decodes the response, and unbinds
in cleanup. It is not a physical microphone, speaker or WAN test. Reports under
`.runtime` contain timings and test transcripts, never device tokens. For JWT
mode, set `NAWEI_SIMULATOR_USER_TOKEN` locally and pass the matching `--user-id`.
`check_full_flow.py` and `check_native_streaming.py` run on the backend host and
read internal service keys from its environment. The full-flow cross-user
lifecycle check uses local authentication; use `--skip-robot` with JWT, then
exercise robot turns with `simulate_robot.py` or `check_native_streaming.py`.

The restart check interrupts Gateway connections, so run it after other live tests.
`restart_robot_services.ps1 -Names llm` also supports reloading the LLM service;
startup and restart share the same local infrastructure environment defaults.
Restart now waits for service readiness (including required LLM memory), with
`-TimeoutSeconds 900` by default. A process launch alone is not reported as ready.
Response-policy checks cover result-only answers,
single-sentence answers without a follow-up and detailed explanations with one
exploration question joining two directions (the current response-policy assertions).
For the nickname/color three-turn simulator scenario, use `--memory-report` with
`check_full_flow.py` after dot-sourcing `scripts/project_runtime.ps1` and calling
`Initialize-NaweiLocalEnvironment`. It compares Redis/API history with PostgreSQL
rows and checks nickname/preference recall. This mode is a local acceptance check.

`401` on OTA generally means an old firmware without `X-Device-Key`, or reset NVS
while ownership still exists. `403` indicates disabled access. Binding returns
`404 activation_invalid`, `410 activation_expired`, `409 activation_used`, or
`409 device_already_bound`. `429` requires waiting for the rate window.
Check model readiness after a startup timeout; cold TTS loading can exceed three
minutes. Source logs are in `logs/latest-runtime.txt`'s runtime directory.

To roll back robot ingress, set `XIAOZHI_ENABLED=false` and restart Gateway.
Preserve the device DB for later recovery. This cleanup does not change existing
bindings, clear NVS, flash a device, or migrate model and memory data.
