# Technical notes from Sparse Blobpool discussion

**(AI-generated, beware of imprecisions and hallucinations).**

Between Marios, Bosul, Csaba, Felix and Raúl.

## Random Query Mechanism - Deep Technical Analysis

**Core Mechanism:**
- Samplers request their custody columns (e.g., 8 columns for a single validator) PLUS 1 additional random column from providers
- Goal: Detect providers falsely claiming full blob possession when they only have partial data

**Attack Scenario Analysis:**
The discussion revealed a sophisticated attack pattern:
1. Attacker identifies peers with overlapping custody columns through network observation over time
2. Attacker stores only 8 specific columns (49% of data with erasure coding)
3. Attacker connects to nodes whose custody sets fully overlap with their 8 columns
4. Without random queries: Attacker successfully serves all custody column requests, appearing as legitimate provider
5. With random queries: Attack cost increases - attacker must hold full blob to handle unpredictable requests

**Why Keep Despite Limited Security:**
- **Peer table persistence attack mitigation:** Without random queries, attackers can build static lists of peers they can consistently fool forever. With random queries, attackers cannot pre-compute which nodes they can reliably trick.
- **Peer quality heuristics:** Even if it doesn't prevent sophisticated attacks, it creates ongoing "honesty signals" that feed into local peer scoring/reputation systems
- **Cost-benefit at 1/8 bandwidth overhead:** For single validator node sampling 8 columns, one extra column = 12.5% overhead. This is deemed acceptable for the added network robustness.
- **Complementary to provider threshold:** Works alongside the 2-provider minimum requirement. Provider threshold handles backbone integrity; random queries add peer-level honesty testing.
- **Network heterogeneity benefit:** Small local randomizations across nodes compound into harder-to-predict network behavior, following P2P design principles of layered defenses.

**Implementation Complexity Trade-off:**
Bosul raised concerns about needing to buffer transactions post-announcement to differentiate malicious non-serving from benign drops due to mempool pressure. However, the discussion concluded this buffering/penalty mechanism is needed anyway for general mempool health (preventing announcement spam), so it's not additional complexity unique to random queries.

---

## Availability Signaling & Worldview Modeling

**Bit Matrix Data Structure:**
- **Dimensions:** Rows = peer IDs, Columns = blob columns/cells
- **Values:** Binary (0/1) or ternary (unknown/partial/full)
- **Timestamps:** Per-cell or per-row decay tracking for confidence degradation
- **Memory overhead:** Acknowledged as significant but manageable given existing mempool overhead

**Signal Types & Interpretation:**

1. **Provider announcements:**
   - Signals "I have full blob" (all 128 columns of extended matrix)
   - Current protocol: node waits for ≥2 provider announcements before acting
   - Rationale: Single provider can be Sybil attack; 2 providers raises attack cost but still relatively weak (attacker can inject 2 colluding nodes)

2. **Sampler announcements:**
   - After fetching custody columns, samplers reannounce
   - Signals "I have these specific columns available"
   - Enables partial availability reconstruction if enough samplers announce

**Phases of Availability:**

**Diffusion/Propagation Phase (first seconds/minutes):**
- High-urgency, bursty traffic
- Goal: Maximize reach and speed
- Don't penalize non-response (peers might be in-flight fetching)
- Don't drop based on lack of signals yet
- Populate worldview opportunistically

**Permanence/Buffering Phase (minutes to hours):**
- Transaction aging in mempool awaiting inclusion
- Goal: Maintain availability confidence, manage space
- Apply multi-factor scoring for retention decisions
- Retest availability periodically

**Retesting Mechanism Details:**
- **Triggering:** Decay functions on timestamps; periodic idle-time sweeps
- **Sampling rate:** Low for regular nodes (minimize overhead), higher for upcoming proposers (need high confidence before building block)
- **Targets:** Randomly sample from worldview to update confidence; prioritize older/lower-confidence entries
- **Actions:** Update bit matrix, adjust retention scores, potentially trigger reconstruction for high-value aged transactions

**Interaction with Peer Churn:**
- New peers join but don't announce existing mempool items
- Solution: Ongoing gossip of announcements for retained blobs (not just initial announcement)
- Allows worldview to stay current as peer table evolves
- Low overhead: Only announcements (hash + bitmap), not full data

---

## Transaction Retention & Dropping - Multi-Factor Scoring

**Attack Vector Being Addressed:**
Bosul's documented attack:
1. Attacker submits high-fee blob transaction
2. Ensures only partial availability (e.g., victim node samples 8 columns, attacker controls those)
3. No other nodes have it (failed propagation)
4. Victim retains large blob in mempool indefinitely
5. Repeat until victim's mempool is full of unavailable garbage
6. Legitimate transactions can't enter

**Eviction Scoring Factors:**

1. **Sampling vs Provider status:** "Did I get full blob or only sampled?" - Higher eviction priority if only sampled
2. **Size:** Larger blobs consume more space - weight by bytes
3. **Age:** Older transactions less likely to be included - time-weighted penalty
4. **Peer availability signals:** From bit matrix - "How many peers announced they have this?" - Few/zero announcements = higher eviction priority
5. **Retest success rate:** For aged transactions, track recent retest results - Consistent failures = higher eviction priority

**Scoring Formula (conceptual):**
```
eviction_score =
  (only_sampled_weight * is_sampled) +
  (size_weight * blob_size_mb) +
  (age_weight * hours_in_mempool) -
  (peer_availability_weight * num_peer_announcements) -
  (retest_success_weight * recent_retest_success_rate)
```

Higher score = evict first when at capacity.

**Phase Separation Rationale:**
- **Don't hinder diffusion:** If nodes aggressively drop during propagation burst based on "not enough peers have it yet," you create a self-fulfilling prophecy where blob fails to propagate
- **Allow settling time:** Give 500ms+ for announcements to arrive, fetches to complete, reannouncements to gossip
- **Then apply retention logic:** After diffusion window, apply scoring to manage mempool under pressure

**Reconstruction Considerations:**

**Against reconstruction for fresh transactions:**
- **Attack amplification:** Attacker sends 51% of blob, network spends CPU reconstructing remaining 49% - cheap attack, expensive defense
- **Lazy publisher incentive:** Publishers optimize by sending only >50% knowing network will reconstruct - shifts cost from sender to receivers
- **Proof verification complexity:** Current proposal sends erasure-coded columns but proof system (KZG commitments) designed for original data - reconstruction requires inverting erasure coding then verifying proofs, complex pipeline

**For reconstruction of aged transactions:**
- After transaction has sat in mempool for extended period (hours?)
- Network has "decided" to retain it (valuable fee, etc.)
- Partial availability observed (e.g., 70% of columns available across peer set)
- Worth spending reconstruction CPU to maintain availability
- Rationale: Amortize one-time cost over long retention period; prevent slow availability decay

**Proof Pipeline Resolution:**
Csaba noted CL discussions about pre-sending proofs alongside data to enable reconstruction without re-verification complexity. Not yet implemented but considered solvable.

---

## Content Addressing for RBF (Replace-By-Fee)

**Problem Statement:**
- User submits blob transaction TX1 with fee F1
- Realizes fee too low, wants to replace with TX2 at fee F2 (RBF)
- TX1 and TX2 share most blob data (e.g., 95% overlap, only metadata changes)
- Current approach: Treat as entirely new blob, re-transfer all 128 columns
- Wasteful: 95% redundant data transfer

**Content Addressing Solution:**
- **Hash-based column identifiers:** Each column identified by content hash (e.g., `vhash(column_data)`)
- **IBF (Invertible Bloom Filter) or similar set reconciliation:** Nodes exchange column ID sets to compute delta
- **Delta fetch:** Only transfer columns with different IDs

**Technical Flow:**
1. Node has TX1 with columns C1...C128, IDs H1...H128
2. TX2 announced with columns C1'...C128', IDs H1'...H128'
3. Set reconciliation: `diff = {H1'...H128'} - {H1...H128}` (e.g., only 6 columns changed)
4. Fetch only those 6 columns
5. Reconstruct full TX2 blob from cached TX1 columns + 6 new columns

**Placement and IBF Details:**
- IBF (Invertible Bloom Filter) allows efficient set reconciliation in single round-trip
- Alternative: Simple bitmap exchange followed by targeted fetch
- Content addressing enables stable identifiers across network, supporting LF (Bloom filter) constructs for efficient announcement/discovery

**Where This Fits:**
- EIP-870: Core mempool RBF mechanism
- Sparse blobpool proposal: Can reference/integrate
- Orthogonal to consensus: Purely networking/mempool optimization

**Why Important for Blobpool:**
Blobs are large (up to ~128KB per column × 128 = ~16MB extended matrix). RBF for blob transactions without content addressing = massive redundant transfers. Content addressing makes RBF practical.

---

## Load Balancing & Bandwidth Management

**The Bottleneck Concern:**
Marios raised: If all nodes fetch from first-heard provider, well-connected/low-latency providers get overloaded (50 peers × 16MB = 800MB burst).

**Existing Mitigations in Geth (per Felix/Csaba):**

1. **Wait-before-fetch window:** 500ms delay after announcement before fetching; if someone pushes data to you during that window, skip fetch
2. **Randomized fetch source:** If multiple providers announced, randomly select which to fetch from
3. **Load balancing across peers:** Spread requests across available sources to avoid single-peer saturation
4. **Staged announcements:** Provider can announce to subset of peers first, wait for them to fetch/reannounce, then announce to more - creates gradual fan-out

**Why Not Specify in Protocol:**
- **Implementation diversity = resilience:** Different clients with different heuristics make network harder to attack/predict
- **Local optimization:** Each node best understands its own constraints (bandwidth, CPU, latency) and can optimize accordingly
- **Avoidance of edge cases:** Normative timing/ordering rules create complex failure modes (e.g., "everyone waits for 2 signals but timeout differs = propagation stall")

**Guidance vs Requirements:**
- Spec should document *patterns* and *considerations*
- Implementers aware of load balancing needs, bloom filter announcements, backpressure
- Avoid mandating specific timeouts, thresholds, algorithms

**Provider-Side Controls:**
- Sender knows bandwidth limits
- Can pace announcements: Send to 10 peers, wait 100ms, send to next 10
- Can throttle responses if saturated
- Network self-heals: If provider A saturated, requesters timeout and try provider B

---

## EIP Status & Fork Planning Context

**Fork Scope Philosophy (Geth Perspective):**
- **Pragmatic filter:** Easy to implement + high impact = prioritize
- **June 2026 target:** Achievable only if scope constrained
- **Repricing EIPs:** Core focus (gas costs, economic security)
- **30+ EIPs proposed:** Clearly unrealistic for single fork

**Sparse Blobpool Position:**
- **Included in rankings:** Shows commitment/progress
- **Noted as non-hard-fork:** Doesn't consume fork complexity budget
- **Different implementation team:** Networking/mempool devs vs consensus/EVM devs - parallel work tracks
- **Vibes/showcase section:** Demonstrates ongoing innovation without blocking fork timeline

**Political Dimension (Felix's note):**
Different client teams may have competing priorities, strategic preferences. Geth going with "pragmatic" approach but others may advocate for their preferred EIPs. Rankings will reflect diverse client perspectives.

---

## Backwards Compatibility & Upgrade Path - Technical Requirements

> HEAVY AI HALLUCINATION! We didn't discuss details in the meeting.

**Critical Design Constraint:**
Must avoid network partition where sparse-blobpool nodes can't communicate with legacy nodes, or where one group sees different mempool state than the other.

**Compatibility Layers:**

1. **DevP2P Subprotocol:**
   - Sparse blobpool uses separate subprotocol capability (e.g., `sbp/1`)
   - Legacy nodes don't advertise/negotiate this capability
   - Nodes supporting both: Serve legacy protocol to old peers, new protocol to upgraded peers
   - Graceful degradation: If peer doesn't speak `sbp/1`, fall back to standard transaction relay

2. **Gossip Topic Compatibility:**
   - Announcements must be interpretable by both node types OR use separate topics
   - Option A: Same topic, extended message format with backward-compatible fields
   - Option B: New topic for sparse announcements, old topic still active for full blobs
   - Key: Legacy nodes still participate in relay even if not understanding sparse semantics

3. **Transaction Format:**
   - No consensus-layer blob transaction format changes
   - Consensus still sees same blob transaction structure
   - Only networking/mempool representation differs (columnar storage, sparse propagation)
   - Ensures proposers can build blocks that all nodes validate identically

4. **Mempool State Consistency:**
   - Both node types accept same transactions as valid
   - Fee market, replacement rules, eviction policies may differ (implementation choice)
   - But acceptance criteria must align: If legacy node accepts TX, upgraded node should too
   - Prevents situations where TX propagates to some nodes but rejected by others on protocol grounds

**Rollout Mechanics:**

**Client-by-client deployment:**
- Geth ships sparse blobpool in v1.X.X
- Nodes upgrade at operator discretion
- No flag day, no coordination requirement
- Network supports heterogeneous mix indefinitely

**Incremental adoption benefits:**
- Early adopters: Reduced bandwidth, better efficiency
- Legacy nodes: Continue operating, gradually see benefits as more peers upgrade
- Network-wide: Resilience improves as sparse propagation becomes more common

**No global threshold requirement:**
- Unlike DAS (Data Availability Sampling) on consensus layer which may need minimum participation
- Sparse blobpool provides benefits even at 10% adoption (those 10% save bandwidth with each other)
- 100% adoption not required for security or functionality

**Testing & Validation:**
Felix emphasized need to confirm these properties before claiming non-fork status. Specific checks:
- Mixed-network propagation tests (legacy ↔️ upgraded)
- Mempool consistency tests (same TX accepted/rejected by both types)
- Fork-resistance tests (ensure no consensus divergence from networking changes)
- Peer connection diversity (upgraded nodes maintain legacy peer connections)

---

## Next Steps - Technical Implementation Focus

**EIP Update (Bosul leading):**

**Content to add/modify:**
1. **Remove random query OR clarify retention:**
   - Keep random query mechanism
   - Specify it's complementary to 2-provider threshold
   - Document as resilience mechanism, not primary security

2. **Availability signaling specification:**
   - Define provider announcement semantics (full blob availability)
   - Define sampler reannouncement semantics (column subset availability)
   - Specify 2-provider minimum threshold before node acts
   - Document worldview construction as implementation guidance

3. **Diffusion vs permanence phases:**
   - Define diffusion window (time-based or event-based?)
   - Propagation rules: Don't penalize, don't drop prematurely
   - Permanence rules: Apply retention scoring, enable retesting
   - Clear transition criteria

4. **Retesting mechanism:**
   - Triggering conditions (age, confidence decay, proposer role)
   - Sampling strategy (random columns, rate-limited)
   - Actions on results (update worldview, trigger reconstruction, evict)
   - Proposer-specific parameters (higher confidence requirements)

5. **Multi-factor retention scoring:**
   - Document factors: sampled/provider status, size, age, peer signals, retest results
   - Leave weights as implementation choice
   - Specify that dropping must allow submitter to RBF/resubmit

6. **Announcement buffering & penalties:**
   - Nodes must serve announced transactions for minimum TTL (e.g., diffusion window + buffer)
   - Failure to serve = penalty signal (disconnect, peer score reduction)
   - Exception: If TX evicted under mempool pressure using multi-factor scoring
   - Distinguish malicious non-serving from benign eviction

**Simulator Development (Raúl leading):**

**Objectives:**
1. **Visualize diffusion:**
   - Network graph with nodes
   - Animate announcement propagation
   - Show provider backbone forming
   - Show sampler column fetches
   - Color-code availability state by node

2. **Attack scenario modeling:**
   - Implement Bosul's garbage mempool attack
   - Show impact without mitigations
   - Show mitigations in action (multi-factor scoring, availability signals)
   - Demonstrate random query benefit against static custody column exploitation

3. **Parameter exploration:**
   - Vary provider threshold (1, 2, 5 providers)
   - Vary peer connectivity (D=25, D=50, D=100)
   - Vary sampling parameters (retest rate, confidence decay)
   - Output: Availability confidence over time, bandwidth usage, attack success rates

4. **Visual materials for post:**
   - Animated GIFs of diffusion process
   - Graphs of availability vs time under attack
   - Comparison charts (legacy vs sparse bandwidth usage)

**RBF Content Addressing (Csaba leading, Raúl collaborating):**

**Specification needs:**
1. **Column identifier scheme:**
   - Hash function (SHA256? Keccak256?)
   - Identifier length (32 bytes? Truncated?)
   - Stability guarantees (same column data = same ID across network)

2. **Set reconciliation protocol:**
   - IBF parameters (size, hash functions, error rates)
   - Alternative: Simple bitmap exchange (128 bits)
   - Round-trip optimization (single message reconciliation?)

3. **Delta fetch protocol:**
   - Request format: List of column IDs needed
   - Response format: Column data + proofs
   - Batch optimization (request multiple deltas in one message)

4. **Integration with mempool:**
   - RBF validation: Check delta size vs full size
   - Fee comparison: New fee must exceed old + delta cost
   - Storage: Deduplication of common columns across TX versions

5. **Proof handling:**
   - KZG commitments for delta columns
   - Verification without full blob reconstruction
   - Batch verification optimization

**Placement decision:** Determine if this goes in EIP-870 (mempool improvements) or sparse blobpool EIP (networking optimization). Leaning toward separate EIP that both can reference.

**Backwards Compatibility Confirmation (Team effort):**

**Test scenarios:**
1. **Mixed network propagation:**
   - Legacy sender → sparse receiver
   - Sparse sender → legacy receiver
   - Multi-hop: Legacy → sparse → legacy chain

2. **Mempool consistency:**
   - Same TX submitted to both node types
   - Verify identical acceptance/rejection
   - Verify identical validation errors

3. **RBF interaction:**
   - Legacy node has TX1, sparse node submits TX2 (RBF)
   - Verify legacy node processes replacement correctly
   - Verify no protocol-level incompatibility

4. **Peer management:**
   - Sparse node connects to mixed peer set
   - Verify maintains connections to legacy peers
   - Verify doesn't downrank legacy peers for not speaking `sbp/1`

5. **Edge cases:**
   - What if blob only available on legacy nodes?
   - What if sparse node samples but can't reconstruct?
   - Fallback to full blob request from legacy peer?

**Documentation:**
- Explicit compatibility statement in EIP
- Test vectors for mixed-network scenarios
- Migration guide for node operators
- Monitoring recommendations (track adoption rate, bandwidth savings)

**Post/Outreach (Marios drafting, Raúl providing visuals):**

**Target audience:**
- Core devs (implementation complexity, benefits)
- Researchers (security analysis, game theory)
- Community (UX improvements, cost reduction)

**Content structure:**
1. **Problem statement:** Current blob propagation is bandwidth-inefficient, full blob to every node
2. **Solution overview:** DAS-inspired columnar sampling with provider backbone
3. **Technical deep-dive:** Custody columns, provider/sampler roles, availability signaling
4. **Security analysis:** Attack scenarios, mitigations, layered defenses
5. **Visual demonstrations:** Simulator animations showing diffusion, attack mitigation
6. **Deployment plan:** Non-fork, gradual rollout, backwards compatible
7. **Call for feedback:** Open questions, parameter choices, implementation concerns
