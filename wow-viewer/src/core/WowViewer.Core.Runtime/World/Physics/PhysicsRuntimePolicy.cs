using WowViewer.Core.Maps;

namespace WowViewer.Core.Runtime.World.Physics;

/// <summary>Whether the selected client build has an evidence-backed physicalised-model path.</summary>
public enum PhysicsAvailability
{
    Unknown = 0,
    Disabled = 1,
    Enabled = 2,
}

/// <summary>Why a candidate did or did not receive a simulation slot for the current frame.</summary>
public enum PhysicsAdmissionReason
{
    Admitted = 0,
    PhysicsDisabled = 1,
    KnownDisabledBuild = 2,
    UnknownBuild = 3,
    BeyondCullDistance = 4,
    ActiveObjectBudgetExhausted = 5,
}

/// <summary>
/// Provenance carried with every build decision. This is deliberately independent of a solver or
/// an unverified physics sidecar layout.
/// </summary>
public readonly record struct PhysicsEvidence(
    string ProfileId,
    string BuildIdentity,
    string EvidenceSource);

/// <summary>An era-scoped physics capability decision for a client build.</summary>
public readonly record struct PhysicsEraDecision(
    PhysicsAvailability Availability,
    PhysicsEvidence Evidence,
    string Diagnostic)
{
    public bool CanAdmitCandidates => Availability == PhysicsAvailability.Enabled;
}

/// <summary>
/// Resolves only build behavior supported by current evidence. Unknown builds fail closed rather than
/// inheriting MoP behavior. The enabled profile is intentionally exact-build until another build is
/// measured.
/// </summary>
public static class PhysicsEraProfileResolver
{
    public const string Alpha053Build = "0.5.3.3368";
    public const string Mop501Build = "5.0.1.15464";
    public const string EvidenceSource = "memory-bank/workstream-atmosphere-501-ghidra.md";

    public static PhysicsEraDecision Resolve(string? buildIdentity)
    {
        string requestedBuild = string.IsNullOrWhiteSpace(buildIdentity)
            ? "<missing>"
            : buildIdentity.Trim();
        if (!ClientBuildKey.TryParse(requestedBuild, out ClientBuildKey build))
        {
            return Unknown(requestedBuild, "The client build identity is missing or malformed.");
        }

        string canonicalBuild = build.ToVersionString();
        if (build == ClientBuildKey.FromVersion(Alpha053Build))
        {
            return new PhysicsEraDecision(
                PhysicsAvailability.Disabled,
                new PhysicsEvidence("physics-alpha-0.5.3-disabled-v1", canonicalBuild, EvidenceSource),
                "Domino physicalised-model behavior is not enabled for the measured alpha profile.");
        }

        if (build == ClientBuildKey.FromVersion(Mop501Build))
        {
            return new PhysicsEraDecision(
                PhysicsAvailability.Enabled,
                new PhysicsEvidence("physics-mop-5.0.1.15464-v1", canonicalBuild, EvidenceSource),
                "The measured 5.0.1 profile permits physics admission; sidecar parsing and simulation remain separate capabilities.");
        }

        return Unknown(canonicalBuild, $"No physics evidence profile exists for build '{canonicalBuild}'.");
    }

    private static PhysicsEraDecision Unknown(string buildIdentity, string diagnostic) => new(
        PhysicsAvailability.Unknown,
        new PhysicsEvidence("physics-unknown-v1", buildIdentity, EvidenceSource),
        diagnostic);
}

/// <summary>Validated, solver-independent controls for physics candidate admission.</summary>
public readonly record struct PhysicsBudget
{
    public PhysicsBudget(bool isEnabled, float cullDistance, int maximumActiveObjects)
    {
        if (!float.IsFinite(cullDistance) || cullDistance < 0)
            throw new ArgumentOutOfRangeException(nameof(cullDistance), "Cull distance must be finite and non-negative.");
        if (maximumActiveObjects < 0)
            throw new ArgumentOutOfRangeException(nameof(maximumActiveObjects), "Maximum active objects must be non-negative.");

        IsEnabled = isEnabled;
        CullDistance = cullDistance;
        MaximumActiveObjects = maximumActiveObjects;
    }

    public bool IsEnabled { get; }

    public float CullDistance { get; }

    public int MaximumActiveObjects { get; }
}

/// <summary>A solver-independent candidate. Lower priority values win, then distance and stable id break ties.</summary>
public readonly record struct PhysicsCandidate(string StableId, float Distance, int Priority = 0);

/// <summary>One explicit, provenance-carrying admission outcome.</summary>
public readonly record struct PhysicsAdmissionDecision(
    string StableId,
    PhysicsAdmissionReason Reason,
    PhysicsAvailability Availability,
    PhysicsEvidence Evidence,
    string Diagnostic)
{
    public bool IsAdmitted => Reason == PhysicsAdmissionReason.Admitted;
}

/// <summary>
/// Deterministically assigns the finite simulation budget. Every input produces exactly one outcome;
/// no candidate can be silently dropped.
/// </summary>
public static class PhysicsAdmissionPolicy
{
    public static IReadOnlyList<PhysicsAdmissionDecision> Evaluate(
        IReadOnlyList<PhysicsCandidate> candidates,
        PhysicsEraDecision era,
        PhysicsBudget budget)
    {
        ArgumentNullException.ThrowIfNull(candidates);
        ValidateEra(era);

        var stableIds = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < candidates.Count; i++)
        {
            ValidateCandidate(candidates[i], i);
            if (!stableIds.Add(candidates[i].StableId))
            {
                throw new ArgumentException(
                    $"Physics candidate stable id '{candidates[i].StableId}' is duplicated.",
                    nameof(candidates));
            }
        }

        PhysicsAdmissionReason globalRefusal = !budget.IsEnabled
            ? PhysicsAdmissionReason.PhysicsDisabled
            : era.Availability switch
            {
                PhysicsAvailability.Disabled => PhysicsAdmissionReason.KnownDisabledBuild,
                PhysicsAvailability.Unknown => PhysicsAdmissionReason.UnknownBuild,
                PhysicsAvailability.Enabled => PhysicsAdmissionReason.Admitted,
                _ => throw new ArgumentOutOfRangeException(
                    nameof(era),
                    era.Availability,
                    "Unknown physics availability value."),
            };

        var decisions = new PhysicsAdmissionDecision[candidates.Count];
        if (globalRefusal != PhysicsAdmissionReason.Admitted)
        {
            for (int i = 0; i < candidates.Count; i++)
                decisions[i] = Decision(candidates[i], globalRefusal, era, budget);
            return decisions;
        }

        var eligible = new List<(int Index, PhysicsCandidate Candidate)>(candidates.Count);
        for (int i = 0; i < candidates.Count; i++)
        {
            PhysicsCandidate candidate = candidates[i];
            if (candidate.Distance > budget.CullDistance)
            {
                decisions[i] = Decision(candidate, PhysicsAdmissionReason.BeyondCullDistance, era, budget);
                continue;
            }

            eligible.Add((i, candidate));
        }

        eligible.Sort(static (left, right) =>
        {
            int comparison = left.Candidate.Priority.CompareTo(right.Candidate.Priority);
            if (comparison != 0)
                return comparison;
            comparison = left.Candidate.Distance.CompareTo(right.Candidate.Distance);
            if (comparison != 0)
                return comparison;
            return StringComparer.Ordinal.Compare(left.Candidate.StableId, right.Candidate.StableId);
        });

        for (int rank = 0; rank < eligible.Count; rank++)
        {
            (int index, PhysicsCandidate candidate) = eligible[rank];
            PhysicsAdmissionReason reason = rank < budget.MaximumActiveObjects
                ? PhysicsAdmissionReason.Admitted
                : PhysicsAdmissionReason.ActiveObjectBudgetExhausted;
            decisions[index] = Decision(candidate, reason, era, budget);
        }

        return decisions;
    }

    private static PhysicsAdmissionDecision Decision(
        PhysicsCandidate candidate,
        PhysicsAdmissionReason reason,
        PhysicsEraDecision era,
        PhysicsBudget budget) => new(
            candidate.StableId,
            reason,
            era.Availability,
            era.Evidence,
            Diagnostic(reason, era, budget));

    private static string Diagnostic(
        PhysicsAdmissionReason reason,
        PhysicsEraDecision era,
        PhysicsBudget budget) => reason switch
        {
            PhysicsAdmissionReason.Admitted => "Candidate admitted to the active physics budget.",
            PhysicsAdmissionReason.PhysicsDisabled => "Physics admission is disabled by the runtime budget.",
            PhysicsAdmissionReason.KnownDisabledBuild => era.Diagnostic,
            PhysicsAdmissionReason.UnknownBuild => era.Diagnostic,
            PhysicsAdmissionReason.BeyondCullDistance =>
                $"Candidate distance exceeds the configured cull distance of {budget.CullDistance}.",
            PhysicsAdmissionReason.ActiveObjectBudgetExhausted =>
                $"Candidate deferred because the active-object limit of {budget.MaximumActiveObjects} is exhausted.",
            _ => throw new ArgumentOutOfRangeException(nameof(reason), reason, "Unknown physics admission reason."),
        };

    private static void ValidateCandidate(PhysicsCandidate candidate, int index)
    {
        if (string.IsNullOrWhiteSpace(candidate.StableId))
            throw new ArgumentException($"Physics candidate {index} must have a stable id.", nameof(candidate));
        if (!float.IsFinite(candidate.Distance) || candidate.Distance < 0)
            throw new ArgumentOutOfRangeException(nameof(candidate), $"Physics candidate '{candidate.StableId}' has an invalid distance.");
    }

    private static void ValidateEra(PhysicsEraDecision era)
    {
        if (era.Availability is not (PhysicsAvailability.Unknown or PhysicsAvailability.Disabled or PhysicsAvailability.Enabled))
            throw new ArgumentOutOfRangeException(nameof(era), era.Availability, "Unknown physics availability value.");
        if (string.IsNullOrWhiteSpace(era.Evidence.ProfileId))
            throw new ArgumentException("Physics era evidence must have a profile id.", nameof(era));
        if (string.IsNullOrWhiteSpace(era.Evidence.BuildIdentity))
            throw new ArgumentException("Physics era evidence must have a build identity.", nameof(era));
        if (string.IsNullOrWhiteSpace(era.Evidence.EvidenceSource))
            throw new ArgumentException("Physics era evidence must have an evidence source.", nameof(era));
        if (string.IsNullOrWhiteSpace(era.Diagnostic))
            throw new ArgumentException("Physics era decisions must have a diagnostic.", nameof(era));
    }
}
