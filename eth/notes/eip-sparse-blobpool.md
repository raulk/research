# EIP: Sparse Blobpool

---
eip: 9999
title: Sparse Blobpool
description: Introduce sampling to the EL blobpool to scale
author:
discussions-to: https://ethereum-magicians.org/t/TBD
status: Draft
type: Standards Track
category: Networking
created: 2025-10-27
requires: 4844, 7594, 7870
---

## Abstract

This proposal introduces the sparse blobpool, a construction that brings cell-level, custody-aligned sampling in the Execution Layer (EL). For every new type 3 (blob-carrying) transaction, an EL node fetches full blob payloads only with probability p = 0.15, and otherwise it merely samples the blobs, using the same custody assignment as its Consensus Layer (CL) counterpart. For full nodes, this means downloading as little as 1/8 of the data (8 out of 128 cells), so that the average bandwidth consumption compared to the (current) full blobpool is 0.15 + 0.85/8 ~ 0.25, a ~4x reduction. The choice of p = 0.15 balances reducing bandwidth consumption with guaranteeing the full propagation of txs, by ensuring that for each blob tx there exists a large connected backbone of nodes that have the full blob payload. At an individual node level, p = 0.15 translates to 98.6% probability of least 3/50 neighbours holding the full blob payload, only 0.03% chance of total unavailability. The sampling performed with probability 1 - p = 0.85 enables streamlined data availability checks during block validation, as well as enhancing the availability of the data.

## Motivation

As Blob Parameter Only (BPO) forks progressively increase throughput, the full-replication nature of today's EL blobpool will begin dominating bandwidth utilization, causing us to hit [EIP-7870](./eip-7870.md) limits. Furthermore, this traffic will compete, and potentially starve, block and attestation propagation, risking instability and liveness issues. This behavior has already been observed in Fusaka devnets, where the average bandwidth consumption *of a full node* is dominated by the EL blobpool, since column propagation on the CL benefits from sampling.

![](https://notes.ethereum.org/_uploads/BkSJvMp0ge.png)

**Figure 1.** *Breakdown of the average bandwidth consumption (download) of full nodes in Fusaka Devnet 5, for blob count target/max of 22/33 (left) and 48/72 (right). The average bandwidth consumption of the EL is ~4-5x that of the CL.*

While the average bandwidth consumption does not reflect that the load on the EL is spread out over time rather than concentrated in a small time window as it is in the CL, the gap between the EL and CL is quite large (~4-5x), and future upgrades will enable the CL to better spread out data propagation in the slot, leaving the EL blobpool as even more of a bottleneck.

The sparse blobpool mechanism brings sampling to the EL as well, with an anticipated ~4x reduction in average bandwidth consumption for a given blobpool load. In doing so, it preserves the unstructured, stochastic nature of the current blobpool rather than introducing complex sharding architectures, prioritizing simplicity and resilience. Moreover, it preserves the CL's ability to satisfy its own sampling needs with the pre-propagated data in the EL blobpool (through the `getBlobs` Engine API). This is achieved by aligning the EL and CL sampling, in particular by having the EL fetch the cells corresponding to the CL custody set. This preserves a key feature of the blobpool, as it stretches the time window for blob data propagation, offloading this work from the critical path of block validation and leading to smoother and less bursty bandwidth utilization patterns over time.

While the scalability gain may be more modest than with other solutions, we believe this design balances between simplicity, security, and scalability, in order to unlock the next tier of blob throughput without requiring user-facing changes, or deeper architectural redesigns.

## Specification

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "NOT RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119) and [RFC 8174](https://www.rfc-editor.org/rfc/rfc8174).

### devp2p Protocol Changes

TODO: check if this is how it is being implemented now

- Extend the `NewPooledTransactionHashes` and `GetPooledTransactions` messages with a `mask` bytes field whose value is contextual to the transaction type.
    - For type 3 transactions, we define `mask` to be the concatenation of a single bitmap of size (128 bits × blob count) and is interpreted as the concatenation of one 128-element bitmap per blob (in transaction order), where 1 bits denote that the cell at that index is available.
    - For other transaction types, `mask` MUST be an RLP `nil` value.
- Extend the `PooledTransactions` message schema to support cell-level returns
- Introduce eth/70 protocol version

### Execution Client Behavior

**New transaction hash.** Upon receiving a `NewPooledTransactionHashes` announcement containing a previously unknown type 3 transaction hash, the node makes a probabilistic decision about fetching the blob payload: it decides to fetch the full blob payload with probability p = 0.15 (provider role), or simply sample them otherwise (sampler role).

**Provider role.** If the node is a provider for a given tx and the announcing peer signaled full availability, the node  MUST request the full transaction payload from that peer. Upon successful retrieval and validation (as per [EIP-7594](./eip-7594.md)), the node MUST in turn announce the transaction hash to its peers via `NewPooledTransactionHashes`.

**Sampler role.** If the node is a sampler, it MUST request the cells corresponding to its custody columns from peers that announced overlapping availability, but only after it observes at least 2 distinct provider announcements. The node MUST only request a maximum of `C_req` columns per request. When fetching from a provider, the node MUST request `C_extra` random columns in addition to its custody set.

**Detecting misbehaviour.** The node MAY keep a record of the frequency of full payload and column requests made by each peer. If a frequency exceeds some quota or the probabilistic expectation by some tolerance threshold, the node MAY decide to disconnect the offending peer alleging abuse as a reason.

**Supernodes.** Supernodes will naturally want to fetch full blob payloads for every transaction. They MUST NOT act like permanent providers, to avoid getting banned due to their fetch rate exceeding expectations. Instead, they MUST respect p = 0.15 and MUST maximally load balance fetching columns from multiple peers. They can remain connected with a larger peerset to satisfy their increased sampling needs.

### Engine API Extensions

**Method `engine_blobCustodyUpdatedV1`**

Called by the Consensus layer client to inform the Execution layer of the indices of their current blob column custody set at startup, as well as subsequent changes during live operation.

Request:
- method: `engine_blobCustodyUpdatedV1`
- params:
    - `indices_bitarray`: uint128, interpreted as a bitarray of length `CELLS_PER_EXT_BLOB` indicating which column indices are currently custodied.
- timeout: 150ms

Response:
- result: no payload
- error: code and message set in case an error occurs during processing of the request.

**Method `engine_getBlobsV4`**

Called by the Consensus layer client to retrieve blob cells from the Execution layer blobpool.

Request:
- method: `engine_getBlobsV4`
- params:
    - `versioned_blob_hashes`: []bytes32, an array of blob versioned hashes.
    - `indices_bitarray`: uint128, a bitarray denoting the indices of the cells to retrieve.
- timeout: 500ms

Response:
- result: `[]BlobCellsAndProofsV1`
- error: code and message set in case an error occurs during processing of the request.

**Data structure `BlobCellsAndProofsV1`**

- `blob_cells`: a sequence of byte arrays `[]bytes` representing the partial matrix of the requested blobs
- `proofs`: `Array of DATA` - Array of `KZGProof` as defined in [EIP-4844](./eip-4844.md), 48 bytes each (`DATA`)

### Parameters

- **Fetching probability**: p = 0.15
- **Mesh degree**: D = 50 (default peerset size)
- **Sampling requirement**: Minimum `SAMPLES_PER_SLOT = 8` columns per node as per PeerDAS specification
- **Reconstruction threshold**: 64 cells required for Reed-Solomon decoding
- **Minimum providers to sample**: minimum 4 peers should be providers in order to sample

## Rationale

### Parameter selection

The choice of p = 0.15 balances bandwidth reduction with availability guarantees. Mathematical analysis for a mesh degree D = 50 shows:

- **Primary reliability**: Probability of having at least 3 peers with complete blob payload is 98.6%
- **Secondary reliability**: Via reconstruction from partial availability, recovery probability exceeds 80% when 6+ provider peers exist.
- **Total unavailability**: Only 0.03% chance with these parameters.

### Reliability framework

**Primary reliability.** Let $X$ be the number of direct peers with the full payload. With $X∼Binomial(D,p)$, the probability that at least $k$ honest peers hold the full blob payload for a type 3 tx is:

$$
P(X \geq k) = 1 - \sum_{i=0}^{k-1} \binom{D}{i} p^i (1-p)^{D-i}
$$

Evaluating for sensible values of $p$ and $k$ yields, where $D=50$ (Geth's default mesh degree):

| $p$ | $k = 6$ | $k = 5$ | $k = 4$ | $k = 3$ | $k = 2$ | $k = 1$ | $P(0)$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.037776 | 0.103617 | 0.239592 | 0.459467 | 0.720568 | 0.923055 | 0.076945 |
| 0.08 | 0.208126 | 0.371050 | 0.574704 | 0.774026 | 0.917288 | 0.984534 | 0.015466 |
| 0.10 | 0.383877 | 0.568802 | 0.749706 | 0.888271 | 0.966214 | 0.994846 | 0.005154 |
| 0.125 | 0.606513 | 0.765366 | 0.886232 | 0.958237 | 0.989739 | 0.998740 | 0.001260 |
| **0.15** | **0.780647** | **0.887895** | **0.953953** | **0.985811** | **0.997095** | **0.999704** | **0.000296** |
| 0.20 | 0.951973 | 0.981504 | 0.994344 | 0.998715 | 0.999807 | 0.999986 | 0.000014 |


**Secondary reliability.** Probability that a payload can be reconstructed from partial availability when primary reliability fails. Let $Y$ be the number of distinct columns available from sampler peers. Given $k$ provider peers that failed to serve the full payload, a node with $D - k$ sampler peers (each holding 8 random columns, assuming they're all minimal custody full nodes) can reconstruct with probability $P(Y \geq 64 \mid n = D - k)$. For the adversarial scenario where we attained $k = 3$, yet all failed to serve the blob data, with $D = 50$, secondary reliability exceeds 99.9% as samplers provide an expected 124 distinct columns from 47 peers.

| Providers ($k$) | Samplers ($n=D-k$) | $E[Distinct Columns]$ | $P(Y ≥ 64)$ |
|---------------|-------------------|---------------------|-----------|
| 0 | 50 | 125.1 | >99.99% |
| 1 | 49 | 124.9 | >99.99% |
| 2 | 48 | 124.6 | >99.99% |
| 3 | 47 | 124.3 | >99.99% |
| 4 | 46 | 124.0 | >99.99% |
| ... | ... | ... | ... |
| ~40 | ~10 | ~68 | ~80% |

**Minimum threshold**: Approximately $n_{\min} \approx 10$ samplers needed for reasonable reconstruction probability (>80%).

## Backwards Compatibility

This EIP changes the `eth` protocol and requires rolling out a new version, `eth/70`. Supporting multiple versions of a wire protocol is possible. Rolling out a new version does not break older clients immediately, since they can keep using protocol version `eth/69`.

This EIP does not change consensus rules of the EVM and does not require a hard fork.

TODO: discuss how a gradual rollout can work

## Test Cases

TBD

## Security Considerations

### DoS

An important consideration in all mempool sharding mechanisms is the possibility of DoS attacks. This refers to the case where a malicious sender posts a transaction to the mempool disclosing only part of it, which makes the transaction impossible to be included in any future block, while still consuming mempool resources.  In this system, such a sender can be detected by nodes that request the full payload of the transaction.

### Peer disconnection policies

Nodes MAY keep a record of the frequency of full payload and column requests made by each peer. If a frequency exceeds some quota or the probabilistic expectation by some tolerance threshold, the node MAY decide to disconnect the offending peer alleging abuse as a reason. This prevents nodes from spending too much upload bandwidth on peers that fetch full payloads much more often that the expected p = 0.15.

## Copyright

Copyright and related rights waived via [CC0](../LICENSE.md).
