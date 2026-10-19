# Adaptive suit deployment closure

## Authority order

Physical ROM validation is not the final product authority boundary. A suit can fit the wearer and reproduce predicted fields while still having unsafe sensor, actuator, power, controller, or field-deployment behavior.

The adaptive chain is therefore:

```text
PhysicalROMValidationReceipt
  -> AdaptiveModuleReceipt
  -> ControllerSafetyReceipt
  -> SuitDeploymentReceipt
```

The implementation lives in:

```text
smii.seams.adaptive_suit_receipt
smii.orchestrator.adaptive_suit_dag
scripts/emit_adaptive_suit_receipts.py
```

## AdaptiveModuleReceipt

The hardware-module receipt hash-binds:

```text
physical ROM receipt
module manifest
sensor map
actuator map
power budget
fail-safe evidence
```

and measures:

```text
module/sensor/actuator inventory
command-envelope violation rate
fail-safe test count
fail-safe failure count
```

Missing inventory or tests remain diagnostic. Command-envelope violations above the declared threshold or any observed fail-safe failure reject.

Safe claim:

> the declared adaptive hardware stack passed the declared hardware-envelope and fail-safe evidence gates.

Unsafe claim:

> the controller is therefore safe, or field deployment is authorized.

## ControllerSafetyReceipt

Controller authority is separate. The receipt binds the promoted adaptive hardware receipt to:

```text
controller artifact
safety envelope
scenario manifest
controller test evidence
```

and records:

```text
scenario count
hard safety violations
maximum command latency / threshold
emergency-stop tests / failures
```

Any hard safety violation, latency violation, or emergency-stop failure rejects. Missing scenario or emergency-stop evidence remains diagnostic.

This is scenario-relative evidence, not a theorem of universal closed-loop safety.

## SuitDeploymentReceipt

Deployment is intentionally not derivable from the preceding proof/test chain. It additionally requires:

```text
deployment profile
deployment evidence
external authority-review receipt
external authority-review decision
```

If external review is missing, deployment remains diagnostic. An explicit review rejection is a hard rejection. A deployment receipt can promote only when the controller receipt is promoted *and* the separate authority review passes.

This prevents the formal/runtime system from silently converting:

```text
"tests passed"
```

into:

```text
"therefore deploy on a human / in the field"
```

without an explicit external authority boundary.

## Full product DAG

The resulting ROM-to-adaptive-suit compiler is now:

```text
BodyCarrierReceipt
  -> CorrespondenceReceipt
  -> BasisReceipt
  -> ROMFieldReceipt
  -> SeamCostReceipt
  -> SolverPromotionReceipt
  -> CutTopologyReceipt
  -> MetricCorrectionReceipt
  -> PanelUnwrapReceipt
  -> ManufacturingReceipt
  -> FinishedSeamReceipt
  -> AssemblyReceipt
  -> WearFitReceipt
  -> PhysicalROMValidationReceipt
  -> AdaptiveModuleReceipt
  -> ControllerSafetyReceipt
  -> SuitDeploymentReceipt
```

`read_adaptive_suit_dag()` checks the hash chain across the last three stages and exposes:

```text
can_validate_controller()
can_request_deployment_authority()
can_field_deploy()
```

The last method reports only receipt eligibility under the declared deployment profile and explicit authority review. It is not a general safety certification.

## Current empirical boundary

No adaptive hardware test dataset, controller scenario corpus, emergency-stop evidence, or deployment authority review is committed by this change. These receipts therefore do not retroactively authorize any physical system.

The immediate upstream empirical blocker also remains unchanged: the current P3/Afflec digital lane must first clear Gate 6. The new depth-N operator-tree backend evaluator is the software route for testing whether the retained P0/P2 search trees improve the real serialization metrics. Only after a pattern promotes do the physical/adaptive receipts become reachable.
