# Sparse blobpool discussion

Bosul, Csaba, Marios, Raúl met to discuss sparse blobpool attack vectors, proposer behavior, and scoring opportunities, sampling noise, and more.

---

## Summary of key points

The discussion centered on analyzing attack vectors for the sparse blob pool, with a major focus on proposer behavior, peer scoring, and a specific nonce-gapping attack.

- **Proposer behavior:** A central debate was whether a proposer must hold the _full blob_ or can propose based on _sampling_. Eagerly fetching the full blob was considered problematic because it de-anonymizes the proposer (revealing them as an upcoming proposer) and would interfere with any statistical modeling of "normal" peer behavior.
- **Peer scoring system (rejected):** The idea of tracking peer statistics (e.g., ratio of full blob requests vs. samples) to detect selfish or malicious nodes was discussed and ultimately **rejected**. The group concluded this mechanism would be too "brittle," hard to enforce correctly, and could unfairly penalize honest proposers who simply want to be certain about blob availability.
- **Sampling noise:** The group largely agreed on adding "sampling noise"—requesting one or more _random_ columns in addition to deterministic custody columns when fetching from a provider. This prevents a simple attack where a malicious node only pretends to be a provider by holding the few columns they know a peer will ask for.
- **Confidence heuristic:** The _true value_ of sampling noise was identified not just as a pass/fail test, but as a tool for proposers. Instead of a binary "sampled/not-sampled" check, a proposer can build a _local confidence score_ based on _how many_ provider announcements they have seen and _successfully tested_ with this random sampling.
- **Nonce-gapping attack:** A localized attack was presented where an attacker eclipses a victim node, tricks it into accepting a transaction (A0), and then spams it with subsequent transactions (A1, A2...) from the same sender. The victim holds A0 (which the rest of the network drops) and thus rejects A1, A2, etc., due to the nonce gap, potentially filling its blob pool.
- **Future solutions:** Unconditional payments and availability certificates were briefly discussed as a more robust, long-term solution to separate blob propagation from availability checks. However, this was deemed a larger, more complex topic (potentially for a future fork) and set aside for a follow-up meeting.

---

## Spec updates & patches required

Based on the discussion, the following items need to be updated, added, or removed from the specification.

1. **Remove peer scoring:** All sections related to statistical tracking, peer quality scoring, or "reasonable behavior" modeling (e.g., tracking the 15% probability) must be **removed**. The system should not be normative about peer request patterns.
2. **Add standard configuration for proposers:** The spec should define a _standard configuration flag_ (likely in the "Rationale" or "Configuration" section for client implementers) that allows a node operator/builder to define their local building policy:
   - **Policy 1 (conservative):** Only propose/include blobs that the node holds _fully_.
   - **Policy 2 (aggressive):** Propose/include blobs that have been _successfully sampled_ (and ideally meet a local confidence threshold).
3. **Implement sampling noise:** The mechanism to request **one (or more) random column(s)** in addition to the deterministic custody columns from a provider must be added.
4. **Define sampling noise failure logic:** The spec must clarify behavior on a sampling noise failure:
   - If a peer _responds_ with custody cells but _fails_ to provide the random cell, it is a clear offense, and the client **should disconnect** from that peer.
   - If the request simply _times out_, it should be handled with more grace (e.g., a few retries) as a general peer quality issue, not an immediate disconnection.
5. **Add blob pool eviction & longevity logic:** To mitigate the nonce-gapping attack and general mempool hygiene, new rules for blob pool management are needed:
   - **Time-to-live (TTL):** Blobs should have a maximum lifetime in the mempool (e.g., a few minutes) and be dropped if not included.
   - **Network saturation check:** A node that has sampled a blob should **drop it** if it does not observe sufficient _network saturation_ (i.e., announcements from other peers for the same blob) within a defined period.
   - **Eviction priority (nonce gaps):** When the blob pool is full, the eviction logic must **prioritize dropping transactions that create nonce gaps** (e.g., A1, A2) over the transaction causing the gap (A0). This prevents the "stacking" part of the attack.
