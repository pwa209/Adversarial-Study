# Narval execution

Source and compact control records: `/home/pwa209/adversarial-study`.
Data, normalized signals, versioned environments and bulk outputs: `/scratch/pwa209/adversarial-study`.

`package_release.py` creates a source-hashed archive and a credential-free deployment request for the established Pertubational Consciousness Manifold authenticated Narval bridge. It creates a new source/run version and submits one finite CPU controller. It does not touch other project jobs or roots.

The controller enforces sixteen allocated CPU cores including itself, plus at most two A100 GPUs. Pending accepted allocations count toward those limits. Worker completion receipts must agree with the accepted job ID and immutable plan hash; the scheduler must also report terminal status before dependencies/resources are released. Ambiguous submission, failed stages and source changes remain explicit. No score/effect/significance is a prerequisite.

Narval compute nodes lack direct public Internet access. `transfer.py run --plan PLAN` is a bounded sequential HTTPS file-management helper on the authenticated login host, at niceness19; it performs no model, preprocessing or scientific-controller work. The Slurm controller waits for a source/plan-bound completion marker, then schedules `transfer.py verify --plan PLAN --group ID` to rehash the actual cache before conversion. There is one kernel-locked transfer helper per study, finite transfer and waiting budgets, and preserved failed/partial receipts. The deployment launcher records its PID separately from Slurm job IDs.

Inspect `runs/<tag>/controller_submission.receipt.json`, `execution_plan.json`, `execution_state.json`, `logs/` and `receipts/`. Numerical and source-specific receipts are at the actual paths in the plan. The controller has a finite allocation: if its walltime expires, first inspect accepted job IDs and submission journals, then resume the same plan in a new one-core Slurm allocation. Never remove started/claim/completion receipts or blindly replay a submitted job. A revised source or scientific specification requires a new release/run version.

The first source-order Monash subject is a runtime pilot, not independent validation. Whole-source development is automatic after numerical/data integrity qualification, even if the pilot contains no desired phenomenon. Fresh independent evaluation and diagnostic collection remain scientific inputs, not pretend-completed scheduler stages. The scientific review at the project root explains the inference boundaries.

## Dependency cache and recovered native import failures

This revision reuses only the exact completed dependency installations from release `20261004T193709Z-545c74b08e71`. Source-pinned previous completion, package-resolution and runtime receipts qualify the cache; each new run checks fresh imports, resolution, active environment and allocated devices. The current source tests run again. Scientific outputs, learned weights, participant partitions and evaluations receive new immutable destinations. Cache mismatch fails closed; there is no silent installation fallback.

The previous release's real MATLAB logs exposed CSV header inference before DREAM and simulated-signal inference. Native batch, human-training and adapted-activation readers now specify comma-delimited headers and required columns explicitly. Original checkpoint and released preprocessing bytes remain fixed; actual native reruns provide execution qualification.
