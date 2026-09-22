# CI result reporting

Use this card when a downstream CI system reports PyTorch compatibility results
to a dashboard or relay.

## Send stable identity

| Field | Why it matters |
|---|---|
| tested PyTorch SHA | associates result with the change or nightly |
| run ID and attempt | separates a rerun from an older result |
| stable job name | lets a dashboard group comparable jobs |
| status and conclusion | distinguishes pending from final outcomes |
| workflow URL | gives an author primary debugging evidence |

For scheduled work, use the upstream PyTorch SHA as the delivery identity—not
the downstream repository commit. A telemetry delivery failure should be
observable, but it should not cause a successful build/test job to be reported
as failed or vice versa.

See the [CRCR module](../../40_crcr_downstream_ci/) for examples.
