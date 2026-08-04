# Paper A measured cost results

> Snapshot: 2026-08-01. Four profiles, 20 validation items each, measured on one H100.
> Source artifacts: `/mnt/persist/paper_a/harvest/cost/*.json`.
> Latencies are CUDA-event medians. Peak memory is incremental allocated memory for the measured phase,
> not total process memory.

## Results

| base | task | median source tokens | median memory tokens | source / memory | encoder ms | compressed read ms | compressed total ms | raw read ms | compressed read peak MiB | raw read peak MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | BFCL | 167.5 | 128 | 1.31× | 33.7 | 76.3 | 110.1 | 47.2 | 49.5 | 59.8 |
| Qwen3.5-9B | BFCL | 170.0 | 128 | 1.33× | 249.4 | 439.3 | 688.7 | 424.5 | 100.0 | 113.7 |
| Qwen3-8B | QuALITY | 4,141.5 | 192 | 23.98× | 165.7 | 82.0 | 247.7 | 200.1 | 74.0 | 996.2 |
| Qwen3.5-9B | QuALITY | 4,239.0 | 192 | 22.79× | 746.4 | 443.3 | 1,189.7 | 1,052.1 | 125.4 | 1,051.1 |

`raw read ms` and peak memory use the 4,096-token configured raw-budget measurement. On sampled
QuALITY items, the actual median source length is slightly above that budget. The 8,192-budget raw medians
are 206.3 ms for Qwen3-8B and 1,128.0 ms for Qwen3.5-9B.

## Interpretation

1. **Reader-state reduction is real on QuALITY.** The compressed reader processes a median 192 memory
   tokens rather than approximately 4.1k raw tokens, reducing incremental read-phase peak allocation from
   about 996/1,051 MiB to 74/125 MiB.
2. **End-to-end speedup is not demonstrated.** Adding the query-conditioned encoder makes the compressed
   path slower than the raw read in all four profiles.
3. **BFCL is not a meaningful compression workload at K=128.** Median source length is only 168–170
   tokens, so the state reduction is approximately 1.3×. Encoder plus compressed-read latency is 1.6–2.3×
   the raw-read latency.
4. **QuALITY exposes the intended systems trade-off.** Reader-side memory is reduced by roughly 23× and
   read-phase allocation falls sharply, but encoder cost erases the latency benefit for a single query.
5. **Reuse is not currently a general answer.** GCM is query-conditioned and normally rebuilds memory for
   a different query. Amortized-speed claims require a separate multi-query or query-agnostic experiment.

## Claim boundary

Supported:

- GCM substantially reduces reader-state length on long documents.
- The compressed read phase uses much less incremental memory on QuALITY.

Not supported:

- GCM provides an end-to-end latency speedup for one query.
- BFCL demonstrates meaningful long-context compression efficiency.
- Query-conditioned memory construction is amortized across unrelated queries.

The manuscript should report encoder, compressed-read, and raw-read phases separately and avoid calling
reader-state reduction an end-to-end speedup.
