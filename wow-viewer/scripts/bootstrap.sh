#!/usr/bin/env bash
set -euo pipefail

include_optional=0
update_existing=0

for arg in "$@"; do
	case "$arg" in
		--include-optional)
			include_optional=1
			;;
		--update-existing)
			update_existing=1
			;;
		*)
			echo "Unknown argument: $arg" >&2
			exit 1
			;;
	esac
done

if ! command -v git >/dev/null 2>&1; then
	echo "Required command 'git' was not found on PATH." >&2
	exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
libs_root="$repo_root/libs"
mkdir -p "$libs_root"

baseline_repos=(
	"wowdev/wow-listfile|https://github.com/wowdev/wow-listfile.git"
	"wowdev/WoWDBDefs|https://github.com/wowdev/WoWDBDefs.git"
	"wowdev/DBCD|https://github.com/wowdev/DBCD.git"
	"ModernWoWTools/Warcraft.NET|https://github.com/ModernWoWTools/Warcraft.NET.git"
	"Marlamin/WoWTools.Minimaps|https://github.com/Marlamin/WoWTools.Minimaps.git"
	"WoW-Tools/SereniaBLPLib|https://github.com/WoW-Tools/SereniaBLPLib.git"
)

optional_repos=(
	"ModernWoWTools/ADTMeta|https://github.com/ModernWoWTools/ADTMeta.git"
	"Marlamin/wow.tools.local|https://github.com/Marlamin/wow.tools.local.git"
	"Kruithne/wow.export|https://github.com/Kruithne/wow.export.git"
	"reference/MapUpconverter|https://github.com/akspa0/MapUpconverter.git"
)

sync_repo() {
	local relative_path="$1"
	local repo_url="$2"
	local target_path="$libs_root/$relative_path"

	mkdir -p "$(dirname "$target_path")"

	if [[ -d "$target_path/.git" ]]; then
		if [[ "$update_existing" -eq 1 ]]; then
			echo "Updating $relative_path"
			git -C "$target_path" pull --ff-only
		else
			echo "Exists   $relative_path"
		fi
		return
	fi

	echo "Cloning  $relative_path"
	git clone --depth 1 "$repo_url" "$target_path"
}

echo "Bootstrapping wow-viewer baseline dependencies"
for entry in "${baseline_repos[@]}"; do
	relative_path="${entry%%|*}"
	repo_url="${entry#*|}"
	sync_repo "$relative_path" "$repo_url"
done

if [[ "$include_optional" -eq 1 ]]; then
	echo "Bootstrapping optional evaluation dependencies"
	for entry in "${optional_repos[@]}"; do
		relative_path="${entry%%|*}"
		repo_url="${entry#*|}"
		sync_repo "$relative_path" "$repo_url"
	done
fi

echo
echo "Baseline bootstrap complete."
echo "Next step: dotnet restore ./WowViewer.slnx"
