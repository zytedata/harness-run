# Scoped GCS token renewal

The client carries the initial token's expiry alongside the bearer on both cold
and warm dispatch paths and seeds the refresh object before starting a worker.
The worker uses that expiry to refresh proactively while its current bearer can
still read the replacement. Client storage calls use the engine's selected
credentials; worker reads use only the scoped bearer.

Older clients omit expiry. Updated workers give these credentials a short retry
schedule instead of treating them as non-expiring. On cold dispatch the new
expiry directive follows the token directive so older token-aware workers still
recognize the token. Older workers will not gain the expiry fix until redeployed;
the unrecognized expiry line may appear in their prompt. Deploy updated workers
before relying on long-turn renewal.

This does not recover a bearer that has already expired, keep refreshing after
the client exits, or prove live platform delivery. Refresh failures retain the
current scoped token and never fall back to ambient credentials. Test clocks,
fake stores and mocked dispatch exercise the wiring offline. A controlled
cold/warm deployment test across a renewal interval remains necessary before
calling the end-to-end production behavior verified.
