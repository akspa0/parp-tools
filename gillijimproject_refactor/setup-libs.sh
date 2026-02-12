#!/usr/bin/env bash
# setup-libs.sh — Bootstrap external library dependencies for building from source.
# Run this once after cloning the repo. Re-run to update libraries.
#
# Usage:
#   ./setup-libs.sh          # Clone all libraries
#   ./setup-libs.sh --force  # Remove and re-clone all libraries

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$SCRIPT_DIR/lib"
FORCE=false

if [[ "${1:-}" == "--force" ]]; then
    FORCE=true
fi

clone_lib() {
    local name="$1"
    local url="$2"
    local target="$3"
    shift 3
    local init_submodules=false
    local submodule_filter=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --submodules) init_submodules=true; shift ;;
            --filter) shift; submodule_filter+=("$1"); shift ;;
            *) shift ;;
        esac
    done

    local full_path="$LIB_DIR/$target"

    if [[ "$FORCE" == true ]] && [[ -d "$full_path" ]]; then
        echo "  Removing existing $target..."
        rm -rf "$full_path"
    fi

    if [[ -d "$full_path" ]]; then
        echo "  $name already exists at $target — skipping (use --force to re-clone)"
        return
    fi

    echo "  Cloning $name..."
    git clone --depth 1 "$url" "$full_path"

    if [[ "$init_submodules" == true ]]; then
        pushd "$full_path" > /dev/null
        if [[ ${#submodule_filter[@]} -gt 0 ]]; then
            for sub in "${submodule_filter[@]}"; do
                echo "    Init submodule: $sub"
                git submodule update --init --depth 1 "$sub"
            done
        else
            git submodule update --init --depth 1
        fi
        popd > /dev/null
    fi
}

echo ""
echo "=== parp-tools: Library Bootstrap ==="
echo "Target: $LIB_DIR"
echo ""

mkdir -p "$LIB_DIR"

# --- SereniaBLPLib (BLP texture loading) ---
clone_lib "SereniaBLPLib" \
    "https://github.com/WoW-Tools/SereniaBLPLib.git" \
    "SereniaBLPLib"

# --- WoWDBDefs (DBC definition files for DBCD) ---
clone_lib "WoWDBDefs" \
    "https://github.com/wowdev/WoWDBDefs.git" \
    "WoWDBDefs"

# --- wow.tools.local (DBCD library + CascLib) ---
clone_lib "wow.tools.local" \
    "https://github.com/Marlamin/wow.tools.local.git" \
    "wow.tools.local" \
    --submodules --filter "DBCD" --filter "CascLib"

# --- Warcraft.NET (WoW file format library) ---
clone_lib "Warcraft.NET" \
    "https://github.com/ModernWoWTools/Warcraft.NET.git" \
    "Warcraft.NET"

# --- WoWTools.Minimaps (StormLibWrapper) ---
# Skip submodules — SereniaBLPLib cloned separately above, TACT.Net cloned below
clone_lib "WoWTools.Minimaps" \
    "https://github.com/Marlamin/WoWTools.Minimaps.git" \
    "WoWTools.Minimaps"

# --- TACT.Net (required by StormLibWrapper) ---
# StormLibWrapper references ../TACT.Net/TACT.Net/TACT.Net.csproj
clone_lib "TACT.Net" \
    "https://github.com/wowdev/TACT.Net.git" \
    "WoWTools.Minimaps/TACT.Net"

echo ""
echo "=== Done! ==="
echo "You can now build MdxViewer:"
echo "  cd src/MdxViewer"
echo "  dotnet build"
echo ""
