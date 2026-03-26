[CmdletBinding()]
param(
	[switch]$IncludeOptional,
	[switch]$UpdateExisting
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$libsRoot = Join-Path $repoRoot 'libs'

$baselineRepos = @(
	@{ Path = 'wowdev/wow-listfile'; Url = 'https://github.com/wowdev/wow-listfile.git' },
	@{ Path = 'wowdev/WoWDBDefs'; Url = 'https://github.com/wowdev/WoWDBDefs.git' },
	@{ Path = 'wowdev/DBCD'; Url = 'https://github.com/wowdev/DBCD.git' },
	@{ Path = 'ModernWoWTools/Warcraft.NET'; Url = 'https://github.com/ModernWoWTools/Warcraft.NET.git' },
	@{ Path = 'Marlamin/WoWTools.Minimaps'; Url = 'https://github.com/Marlamin/WoWTools.Minimaps.git' },
	@{ Path = 'WoW-Tools/SereniaBLPLib'; Url = 'https://github.com/WoW-Tools/SereniaBLPLib.git' }
)

$optionalRepos = @(
	@{ Path = 'ModernWoWTools/ADTMeta'; Url = 'https://github.com/ModernWoWTools/ADTMeta.git' },
	@{ Path = 'Marlamin/wow.tools.local'; Url = 'https://github.com/Marlamin/wow.tools.local.git' },
	@{ Path = 'Kruithne/wow.export'; Url = 'https://github.com/Kruithne/wow.export.git' },
	@{ Path = 'reference/MapUpconverter'; Url = 'https://github.com/akspa0/MapUpconverter.git' }
)

function Assert-CommandAvailable {
	param([string]$Name)

	if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
		throw "Required command '$Name' was not found on PATH."
	}
}

function Sync-Repository {
	param(
		[string]$TargetPath,
		[string]$RepositoryUrl
	)

	$fullTargetPath = Join-Path $libsRoot $TargetPath
	$parent = Split-Path -Parent $fullTargetPath
	if (-not (Test-Path $parent)) {
		New-Item -ItemType Directory -Force -Path $parent | Out-Null
	}

	$gitDir = Join-Path $fullTargetPath '.git'
	if (Test-Path $gitDir) {
		if ($UpdateExisting) {
			Write-Host "Updating $TargetPath"
			git -C $fullTargetPath pull --ff-only
		}
		else {
			Write-Host "Exists   $TargetPath"
		}

		return
	}

	Write-Host "Cloning  $TargetPath"
	git clone --depth 1 $RepositoryUrl $fullTargetPath
}

Assert-CommandAvailable 'git'

if (-not (Test-Path $libsRoot)) {
	New-Item -ItemType Directory -Force -Path $libsRoot | Out-Null
}

Write-Host 'Bootstrapping wow-viewer baseline dependencies'
foreach ($repo in $baselineRepos) {
	Sync-Repository -TargetPath $repo.Path -RepositoryUrl $repo.Url
}

if ($IncludeOptional) {
	Write-Host 'Bootstrapping optional evaluation dependencies'
	foreach ($repo in $optionalRepos) {
		Sync-Repository -TargetPath $repo.Path -RepositoryUrl $repo.Url
	}
}

Write-Host ''
Write-Host 'Baseline bootstrap complete.'
Write-Host 'Next step: dotnet restore .\WowViewer.slnx'
