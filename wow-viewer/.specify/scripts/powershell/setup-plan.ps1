#!/usr/bin/env pwsh
# Setup implementation plan for a feature

[CmdletBinding()]
param(
    [switch]$Json,
    [switch]$Force,
    [switch]$Help
)

$ErrorActionPreference = 'Stop'

# Show help if requested
if ($Help) {
    Write-Output "Usage: ./setup-plan.ps1 [-Json] [-Force] [-Help]"
    Write-Output "  -Json     Output results in JSON format"
    Write-Output "  -Force    Overwrite an existing plan.md with a blank template (DESTRUCTIVE)"
    Write-Output "  -Help     Show this help message"
    exit 0
}

# Load common functions
. "$PSScriptRoot/common.ps1"

# Get all paths and variables from common functions
$paths = Get-FeaturePathsEnv

# If feature.json pins an existing feature directory, branch naming is not required.
if (-not (Test-FeatureJsonMatchesFeatureDir -RepoRoot $paths.REPO_ROOT -ActiveFeatureDir $paths.FEATURE_DIR)) {
    if (-not (Test-FeatureBranch -Branch $paths.CURRENT_BRANCH -HasGit $paths.HAS_GIT)) {
        exit 1
    }
}

# Ensure the feature directory exists
New-Item -ItemType Directory -Path $paths.FEATURE_DIR -Force | Out-Null

# Seed plan.md from the template, but NEVER destroy authored work.
#
# This previously wrote the template unconditionally. Combined with a stale feature.json pointing
# at a different feature, that silently overwrote a completed plan with an empty template. Writing
# a scaffold is not worth losing authored content, and create-new-feature.ps1 already guards
# spec.md the same way -- this is just the missing half of that pattern.
$planExists = Test-Path -LiteralPath $paths.IMPL_PLAN -PathType Leaf
$planPreserved = $false

if ($planExists -and -not $Force) {
    # Only skip when the existing file actually holds something other than the pristine template.
    $existing = [System.IO.File]::ReadAllText($paths.IMPL_PLAN)
    $templatePath = Resolve-Template -TemplateName 'plan-template' -RepoRoot $paths.REPO_ROOT
    $templateText = if ($templatePath -and (Test-Path $templatePath)) { [System.IO.File]::ReadAllText($templatePath) } else { '' }

    if ($existing.Trim() -and $existing.Trim() -ne $templateText.Trim()) {
        $planPreserved = $true
        Write-Warning "[specify] plan.md already exists and differs from the template; keeping it. Re-run with -Force to overwrite."
    }
}

if (-not $planPreserved) {
    $template = Resolve-Template -TemplateName 'plan-template' -RepoRoot $paths.REPO_ROOT
    if ($template -and (Test-Path $template)) {
        # Read the template content and write it to the implementation plan file with UTF-8 encoding without BOM
        $content = [System.IO.File]::ReadAllText($template)
        $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText($paths.IMPL_PLAN, $content, $utf8NoBom)
    } else {
        Write-Warning "Plan template not found"
        # Create a basic plan file if template doesn't exist
        New-Item -ItemType File -Path $paths.IMPL_PLAN -Force | Out-Null
    }
}

# Output results
if ($Json) {
    $result = [PSCustomObject]@{
        FEATURE_SPEC = $paths.FEATURE_SPEC
        IMPL_PLAN = $paths.IMPL_PLAN
        SPECS_DIR = $paths.FEATURE_DIR
        BRANCH = $paths.CURRENT_BRANCH
        HAS_GIT = $paths.HAS_GIT
        PLAN_PRESERVED = $planPreserved
    }
    $result | ConvertTo-Json -Compress
} else {
    Write-Output "FEATURE_SPEC: $($paths.FEATURE_SPEC)"
    Write-Output "IMPL_PLAN: $($paths.IMPL_PLAN)"
    Write-Output "SPECS_DIR: $($paths.FEATURE_DIR)"
    Write-Output "BRANCH: $($paths.CURRENT_BRANCH)"
    Write-Output "HAS_GIT: $($paths.HAS_GIT)"
    Write-Output "PLAN_PRESERVED: $planPreserved"
}
