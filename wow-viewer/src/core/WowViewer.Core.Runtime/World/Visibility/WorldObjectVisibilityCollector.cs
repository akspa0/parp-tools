using System.Numerics;
using WowViewer.Core.Runtime.World;

namespace WowViewer.Core.Runtime.World.Visibility;

public static class WorldObjectVisibilityCollector
{
    private const float DefaultVerticalFovRadians = MathF.PI / 3f;
    private const float MinVerticalFovRadians = 20f * (MathF.PI / 180f);
    private const float MaxVerticalFovRadians = 90f * (MathF.PI / 180f);
    private const float DoodadCullDistance = 5000f;
    private const float DoodadCullDistanceSq = DoodadCullDistance * DoodadCullDistance;
    private const float DoodadSmallThreshold = 10f;
    private const float FadeStartFraction = 0.80f;
    private const float WmoCullDistance = 1600f;
    private const float NoCullRadius = 512f;
    private const float ObjectNearHoldRadius = 384f;
    private const float ObjectNearHoldRadiusSq = ObjectNearHoldRadius * ObjectNearHoldRadius;
    private const float VisionConeFrontDot = 0.15f;
    private const float VisionConeRearDot = -0.35f;
    private const float RearConeCullFraction = 0.45f;
    private const float MinOffFrustumConeFactor = 0.35f;
    private const float RearConeFadeFloor = 0.25f;
    private const float RearConeLoadPenalty = 2.5f;
    private const float MaxWorldObjectViewDistance = 8192f;
    private const float MaxWorldObjectViewDistanceSq = MaxWorldObjectViewDistance * MaxWorldObjectViewDistance;

    public static int CollectVisibleWmos(
        WorldVisibilityFrame frame,
        IReadOnlyList<WorldObjectInstance> instances,
        WorldObjectVisibilityContext context,
        Func<WorldObjectInstance, bool> shouldHideInstance,
        Func<Vector3, Vector3, bool> isBoundsVisible,
        Func<string, bool> isAssetReady,
        Action<string, float> queuePendingAsset)
    {
        int culledCount = 0;
        float wmoCullDistance = ComputeWmoCullDistance(context.FogEnd, context.ObjectStreamingRangeMultiplier);

        for (int i = 0; i < instances.Count; i++)
        {
            WorldObjectInstance inst = instances[i];
            if (shouldHideInstance(inst))
                continue;

            float boundsDistSq = DistanceSquaredPointToAabb(context.CameraPosition, inst.BoundsMin, inst.BoundsMax);
            float centerDistanceSq = Vector3.DistanceSquared(context.CameraPosition, inst.PlacementPosition);
            float coneFactor = ComputeVisionConeFactor(context.CameraPosition, context.CameraForward, inst.PlacementPosition, centerDistanceSq);
            float coneCullDistance = ComputeConeCullDistance(wmoCullDistance, coneFactor);
            float coneCullDistanceSq = coneCullDistance * coneCullDistance;
            float noCullDistanceSq = ComputeNoCullDistanceSq(inst.BoundsMin, inst.BoundsMax);
            bool frustumVisible = isBoundsVisible(inst.BoundsMin, inst.BoundsMax);
            float projectedFraction = ComputeProjectedHeightFraction(inst.BoundsMin, inst.BoundsMax, centerDistanceSq, context.VerticalFieldOfViewRadians);
            if (boundsDistSq > noCullDistanceSq && !frustumVisible && coneFactor < MinOffFrustumConeFactor)
            {
                culledCount++;
                continue;
            }

            if (boundsDistSq > coneCullDistanceSq)
            {
                culledCount++;
                continue;
            }

            if (centerDistanceSq > MaxWorldObjectViewDistanceSq)
            {
                culledCount++;
                continue;
            }

            if (ShouldCullByProjectedSize(context, projectedFraction, centerDistanceSq, isWmo: true))
            {
                culledCount++;
                continue;
            }

            if (!isAssetReady(inst.ModelKey))
            {
                if (ShouldQueuePendingAsset(context, frustumVisible, coneFactor, projectedFraction, centerDistanceSq, isWmo: true))
                    queuePendingAsset(inst.ModelKey, ComputeLoadPriorityScore(centerDistanceSq, coneFactor));

                continue;
            }

            frame.VisibleWmos.Add(new WorldVisibleWmoEntry(inst, centerDistanceSq));
        }

        return culledCount;
    }

    public static int CollectVisibleMdx(
        WorldVisibilityFrame frame,
        IReadOnlyList<WorldObjectInstance> instances,
        WorldObjectVisibilityContext context,
        Func<WorldObjectInstance, bool> shouldHideInstance,
        Func<Vector3, Vector3, bool> isBoundsVisible,
        Func<string, bool> isAssetReady,
        Action<string, float> queuePendingAsset)
    {
        int culledCount = 0;

        for (int i = 0; i < instances.Count; i++)
        {
            WorldObjectInstance inst = instances[i];
            if (shouldHideInstance(inst))
                continue;

            float boundsDistSq = DistanceSquaredPointToAabb(context.CameraPosition, inst.BoundsMin, inst.BoundsMax);
            float centerDistanceSq = Vector3.DistanceSquared(context.CameraPosition, inst.Transform.Translation);
            float coneFactor = ComputeVisionConeFactor(context.CameraPosition, context.CameraForward, inst.Transform.Translation, centerDistanceSq);
            float noCullDistanceSq = ComputeNoCullDistanceSq(inst.BoundsMin, inst.BoundsMax);
            bool frustumVisible = isBoundsVisible(inst.BoundsMin, inst.BoundsMax);
            if (boundsDistSq > noCullDistanceSq && !frustumVisible && coneFactor < MinOffFrustumConeFactor)
            {
                culledCount++;
                continue;
            }

            float diag = (inst.BoundsMax - inst.BoundsMin).Length();
            float mdxCullDistance = ComputeMdxCullDistance(context.FogEnd, diag, context.CountAsTaxiActor, context.ObjectStreamingRangeMultiplier);
            float coneCullDistance = ComputeConeCullDistance(mdxCullDistance, coneFactor);
            float coneCullDistanceSq = coneCullDistance * coneCullDistance;
            if (boundsDistSq > coneCullDistanceSq)
            {
                culledCount++;
                continue;
            }

            bool useSmallDoodadCull = context.CullSmallDoodadsOnly && diag < DoodadSmallThreshold;
            if (useSmallDoodadCull && boundsDistSq > DoodadCullDistanceSq)
            {
                culledCount++;
                continue;
            }

            if (centerDistanceSq > MaxWorldObjectViewDistanceSq)
            {
                culledCount++;
                continue;
            }

            float projectedFraction = ComputeProjectedHeightFraction(inst.BoundsMin, inst.BoundsMax, centerDistanceSq, context.VerticalFieldOfViewRadians);
            if (ShouldCullByProjectedSize(context, projectedFraction, centerDistanceSq, isWmo: false))
            {
                culledCount++;
                continue;
            }

            if (!isAssetReady(inst.ModelKey))
            {
                if (ShouldQueuePendingAsset(context, frustumVisible, coneFactor, projectedFraction, centerDistanceSq, isWmo: false))
                    queuePendingAsset(inst.ModelKey, ComputeLoadPriorityScore(centerDistanceSq, coneFactor));

                continue;
            }

            float opaqueFade = 1.0f;
            float mdxFadeStart = coneCullDistance * FadeStartFraction;
            float mdxFadeStartSq = mdxFadeStart * mdxFadeStart;
            float mdxFadeRange = MathF.Max(1f, coneCullDistance - mdxFadeStart);
            if (boundsDistSq > mdxFadeStartSq)
            {
                float boundsDist = MathF.Sqrt(boundsDistSq);
                opaqueFade = MathF.Max(0f, 1.0f - (boundsDist - mdxFadeStart) / mdxFadeRange);
            }

            float transparentFade = 1.0f;
            if (centerDistanceSq > mdxFadeStartSq)
            {
                float centerDistance = MathF.Sqrt(centerDistanceSq);
                transparentFade = MathF.Max(0f, 1.0f - (centerDistance - mdxFadeStart) / mdxFadeRange);
            }

            float coneFade = ComputeConeFade(coneFactor, centerDistanceSq);
            opaqueFade *= coneFade;
            transparentFade *= coneFade;

            frame.VisibleMdx.Add(new WorldVisibleMdxEntry(inst, centerDistanceSq, opaqueFade, transparentFade, context.CountAsTaxiActor));
            if (context.CountAsTaxiActor)
                frame.VisibleTaxiMdxCount++;
        }

        return culledCount;
    }

    private static float DistanceSquaredPointToAabb(Vector3 point, Vector3 min, Vector3 max)
    {
        float dx = point.X < min.X ? min.X - point.X : point.X > max.X ? point.X - max.X : 0f;
        float dy = point.Y < min.Y ? min.Y - point.Y : point.Y > max.Y ? point.Y - max.Y : 0f;
        float dz = point.Z < min.Z ? min.Z - point.Z : point.Z > max.Z ? point.Z - max.Z : 0f;
        return dx * dx + dy * dy + dz * dz;
    }

    private static float ComputeNoCullDistanceSq(Vector3 min, Vector3 max)
    {
        float halfDiagonal = (max - min).Length() * 0.5f;
        float graceRadius = MathF.Max(NoCullRadius, MathF.Min(halfDiagonal + 96f, 1024f));
        return graceRadius * graceRadius;
    }

    private static float ComputeWmoCullDistance(float fogEnd, float rangeMultiplier)
    {
        float clampedMultiplier = Math.Clamp(rangeMultiplier, 1.0f, 4.0f);
        if (fogEnd <= 0f)
            return MathF.Min(MaxWorldObjectViewDistance, MathF.Min(WmoCullDistance, MaxWorldObjectViewDistance) * clampedMultiplier);

        float baseDistance = MathF.Min(MaxWorldObjectViewDistance, MathF.Max(WmoCullDistance, fogEnd + 256f));
        return MathF.Min(MaxWorldObjectViewDistance, baseDistance * clampedMultiplier);
    }

    private static float ComputeMdxCullDistance(float fogEnd, float boundsDiagonal, bool isTaxiActor, float rangeMultiplier)
    {
        float clampedMultiplier = Math.Clamp(rangeMultiplier, 1.0f, 4.0f);
        if (isTaxiActor)
            return MathF.Min(MaxWorldObjectViewDistance, MathF.Max(1024f, fogEnd + 384f) * clampedMultiplier);

        if (fogEnd <= 0f)
            return MathF.Min(MaxWorldObjectViewDistance, MathF.Min(DoodadCullDistance, MaxWorldObjectViewDistance) * clampedMultiplier);

        float objectAllowance = MathF.Min(512f, boundsDiagonal * 0.5f + 96f);
        float baseDistance = MathF.Min(DoodadCullDistance, MathF.Max(1024f, fogEnd + objectAllowance));
        return MathF.Min(MaxWorldObjectViewDistance, baseDistance * clampedMultiplier);
    }

    private static float ComputeVisionConeFactor(Vector3 cameraPos, Vector3 cameraForward, Vector3 targetPos, float targetDistanceSq)
    {
        if (targetDistanceSq <= ObjectNearHoldRadiusSq)
            return 1.0f;

        float forwardLengthSq = cameraForward.LengthSquared();
        if (forwardLengthSq <= 1e-6f)
            return 1.0f;

        Vector3 toTarget = targetPos - cameraPos;
        float toTargetLengthSq = toTarget.LengthSquared();
        if (toTargetLengthSq <= 1e-6f)
            return 1.0f;

        float invTargetLength = 1.0f / MathF.Sqrt(toTargetLengthSq);
        float alignment = Vector3.Dot(toTarget * invTargetLength, cameraForward);
        float factor = (alignment - VisionConeRearDot) / MathF.Max(0.001f, VisionConeFrontDot - VisionConeRearDot);
        return Math.Clamp(factor, 0.0f, 1.0f);
    }

    private static float ComputeConeCullDistance(float baseCullDistance, float coneFactor)
    {
        if (baseCullDistance <= 0f)
            return ObjectNearHoldRadius;

        float scale = RearConeCullFraction + (1.0f - RearConeCullFraction) * coneFactor;
        return MathF.Max(ObjectNearHoldRadius, baseCullDistance * scale);
    }

    private static float ComputeConeFade(float coneFactor, float centerDistanceSq)
    {
        if (centerDistanceSq <= ObjectNearHoldRadiusSq)
            return 1.0f;

        return RearConeFadeFloor + (1.0f - RearConeFadeFloor) * coneFactor;
    }

    private static float ComputeLoadPriorityScore(float centerDistanceSq, float coneFactor)
    {
        float penalty = RearConeLoadPenalty - (RearConeLoadPenalty - 1.0f) * coneFactor;
        return centerDistanceSq * penalty;
    }

    private static float ComputeProjectedHeightFraction(Vector3 min, Vector3 max, float centerDistanceSq, float verticalFovRadians)
    {
        if (centerDistanceSq <= ObjectNearHoldRadiusSq)
            return 1.0f;

        float distance = MathF.Sqrt(MathF.Max(centerDistanceSq, 1e-6f));
        float clampedFov = Math.Clamp(verticalFovRadians, MinVerticalFovRadians, MaxVerticalFovRadians);
        float tanHalfFov = MathF.Tan(clampedFov * 0.5f);
        if (!float.IsFinite(tanHalfFov) || tanHalfFov <= 1e-6f)
            tanHalfFov = MathF.Tan(DefaultVerticalFovRadians * 0.5f);

        float diagonal = (max - min).Length();
        if (diagonal <= 1e-6f)
            return 0f;

        return diagonal / MathF.Max(1e-6f, 2f * distance * tanHalfFov);
    }

    private static bool ShouldCullByProjectedSize(
        WorldObjectVisibilityContext context,
        float projectedFraction,
        float centerDistanceSq,
        bool isWmo)
    {
        if (centerDistanceSq <= ObjectNearHoldRadiusSq || context.CountAsTaxiActor)
            return false;

        float threshold = GetRenderProjectedFractionThreshold(context.VisibilityProfile, isWmo);
        return threshold > 0f && projectedFraction < threshold;
    }

    private static bool ShouldQueuePendingAsset(
        WorldObjectVisibilityContext context,
        bool frustumVisible,
        float coneFactor,
        float projectedFraction,
        float centerDistanceSq,
        bool isWmo)
    {
        if (centerDistanceSq <= ObjectNearHoldRadiusSq)
            return true;

        float minProjectedFraction = GetLoadProjectedFractionThreshold(context.VisibilityProfile, isWmo);
        if (projectedFraction < minProjectedFraction)
            return false;

        if (frustumVisible)
            return true;

        return coneFactor >= GetLoadConeFactorThreshold(context.VisibilityProfile, isWmo);
    }

    private static float GetRenderProjectedFractionThreshold(WorldObjectVisibilityProfile profile, bool isWmo)
    {
        return profile switch
        {
            WorldObjectVisibilityProfile.Quality => 0f,
            WorldObjectVisibilityProfile.Balanced => isWmo ? 0.0009f : 0.0020f,
            _ => isWmo ? 0.0014f : 0.0035f,
        };
    }

    private static float GetLoadProjectedFractionThreshold(WorldObjectVisibilityProfile profile, bool isWmo)
    {
        return profile switch
        {
            WorldObjectVisibilityProfile.Quality => 0f,
            WorldObjectVisibilityProfile.Balanced => isWmo ? 0.0011f : 0.0025f,
            _ => isWmo ? 0.0018f : 0.0045f,
        };
    }

    private static float GetLoadConeFactorThreshold(WorldObjectVisibilityProfile profile, bool isWmo)
    {
        return profile switch
        {
            WorldObjectVisibilityProfile.Quality => MinOffFrustumConeFactor,
            WorldObjectVisibilityProfile.Balanced => isWmo ? 0.45f : 0.55f,
            _ => isWmo ? 0.60f : 0.72f,
        };
    }
}