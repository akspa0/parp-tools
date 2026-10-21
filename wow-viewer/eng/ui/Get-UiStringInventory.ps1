#requires -Version 7.0
<#
.SYNOPSIS
Inventories viewer UI string callsites without building or launching the viewer.
.DESCRIPTION
Uses the Roslyn assemblies bundled with PowerShell 7. No packages, SDK build,
client data, or network access are needed. Output is ordinal-sorted UTF-8 TSV
with LF endings. Literal values are decoded; dynamic expressions remain source
text. Backslashes, tabs and line breaks are escaped in every field.

Direct ImGui controls/text, draw-list AddText, and syntactically inferred string
wrapper arguments are included. Wrapper matching is by method name/parameter
shape, NOT semantic binding. Local variable dependencies are followed only to
infer wrapper parameters, not to evaluate dynamic labels. Options, previews,
hints, shortcuts, and non-visible container IDs have separate roles.

Menu paths describe enclosing positive if (ImGui.BeginMenu(...)) branches in
the same syntax tree, not the runtime call graph. Helper-generated menus and
early-return menu patterns can lack ancestors. Reachability, runtime strings,
resource strings, option-array expansion and control functionality are NOT
proved. Conditional code uses the supplied -Define symbols (none by default).

Existing output is protected unless -Force is supplied. -Verify regenerates
in memory and compares bytes without writing. Source hashes are rechecked
before writing to reject a baseline captured during concurrent source edits.
.EXAMPLE
pwsh -NoProfile -File wow-viewer/eng/ui/Get-UiStringInventory.ps1 -OutputPath wow-viewer/specs/280-ui-audit-refactor/evidence/ui-strings-baseline.txt
.EXAMPLE
pwsh -NoProfile -File wow-viewer/eng/ui/Get-UiStringInventory.ps1 -SelfTest
#>
[CmdletBinding(DefaultParameterSetName = 'Inventory')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Inventory')]
    [string]$OutputPath,
    [Parameter(ParameterSetName = 'Inventory')]
    [string]$RepositoryRoot = (Join-Path $PSScriptRoot '../../..'),
    [Parameter(ParameterSetName = 'Inventory')]
    [string]$ViewerPath = 'wow-viewer/src/viewer',
    [string[]]$Define = @(),
    [Parameter(ParameterSetName = 'Inventory')]
    [switch]$Verify,
    [Parameter(ParameterSetName = 'Inventory')]
    [switch]$Force,
    [Parameter(Mandatory, ParameterSetName = 'SelfTest')]
    [switch]$SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Add-Type -Path (Join-Path $PSHOME 'Microsoft.CodeAnalysis.dll')
Add-Type -Path (Join-Path $PSHOME 'Microsoft.CodeAnalysis.CSharp.dll')

function ConvertTo-TsvField([string]$Value) {
    $Value.Replace('\', '\\').Replace("`t", '\t').Replace("`r", '\r').Replace("`n", '\n')
}

function Get-CallName($Call) {
    $expression = $Call.Expression
    if ($expression -is [Microsoft.CodeAnalysis.CSharp.Syntax.MemberAccessExpressionSyntax] -or
        $expression -is [Microsoft.CodeAnalysis.CSharp.Syntax.MemberBindingExpressionSyntax]) {
        return $expression.Name.Identifier.ValueText
    }
    if ($expression -is [Microsoft.CodeAnalysis.CSharp.Syntax.SimpleNameSyntax]) {
        return $expression.Identifier.ValueText
    }
    return ''
}

function New-Slot([int]$Index, [string]$Role, [string]$Names) {
    @{ Index = $Index; Role = $Role; Names = $Names.Split(',') }
}

function Get-DirectSlots($Call) {
    $name = Get-CallName $Call
    if ($Call.Expression -is [Microsoft.CodeAnalysis.CSharp.Syntax.MemberAccessExpressionSyntax] -and
        $Call.Expression.Expression.ToString() -match '(^|\.)ImGui$') {
        switch -Regex ($name) {
            '^(Text|TextDisabled|TextWrapped|TextUnformatted|BulletText|SetTooltip|SetItemTooltip|SeparatorText)$' {
                New-Slot 0 'text' 'fmt,text,text_begin'; break
            }
            '^TextColored$' { New-Slot 1 'text' 'fmt,text'; break }
            '^(BeginChild|BeginTable|BeginTabBar|BeginPopup|BeginPopupContextItem|BeginPopupContextWindow|BeginPopupContextVoid|OpenPopup|InvisibleButton|ArrowButton)$' {
                New-Slot 0 'id' 'str_id,id'; break
            }
            '^ProgressBar$' { New-Slot 2 'overlay' 'overlay'; break }
            '^(Image|ImageWithBg)$' { New-Slot -1 'unlabelled-control' ''; break }
            '^(Begin|BeginMenu|MenuItem|Button|SmallButton|Checkbox|CheckboxFlags|RadioButton|Selectable|CollapsingHeader|BeginCombo|Combo|ListBox|BeginListBox|TreeNode|TreeNodeEx|BeginTabItem|TabItemButton|BeginPopupModal|TableSetupColumn|ImageButton|PlotLines|PlotHistogram|Input(Text|Int|Float|Double|Scalar).*|Slider.*|VSlider.*|Drag.*|Color(Edit|Picker|Button).*)$' {
                New-Slot 0 'label' 'label,name,str_id,tab_label'; break
            }
        }
        switch ($name) {
            'InputTextWithHint' { New-Slot 1 'hint' 'hint' }
            'BeginCombo' { New-Slot 1 'preview' 'preview_value' }
            'MenuItem' {
                # The two-argument overload also accepts an enabled bool. Do
                # not misreport an unresolved boolean expression as UI text.
                $shortcut = Get-SlotArgument $Call (New-Slot 1 'shortcut' 'shortcut')
                if ($Call.ArgumentList.Arguments.Count -ge 3 -or
                    $shortcut -is [Microsoft.CodeAnalysis.CSharp.Syntax.InterpolatedStringExpressionSyntax] -or
                    ($shortcut -is [Microsoft.CodeAnalysis.CSharp.Syntax.LiteralExpressionSyntax] -and
                     $shortcut.Token.Value -is [string])) {
                    New-Slot 1 'shortcut' 'shortcut'
                }
            }
            'Combo' { New-Slot 2 'options' 'items,items_separated_by_zeros,items_getter' }
            'ListBox' { New-Slot 2 'options' 'items,items_getter' }
            'TreeNode' {
                if ($Call.ArgumentList.Arguments.Count -ge 2) { New-Slot 1 'text' 'fmt' }
            }
            'TreeNodeEx' {
                if ($Call.ArgumentList.Arguments.Count -ge 3) { New-Slot 2 'text' 'fmt' }
            }
        }
    }
    elseif ($name -eq 'AddText') {
        # ImDrawListPtr overloads: (pos, col, text) or (font, size, pos, col, text, ...).
        $index = if ($Call.ArgumentList.Arguments.Count -ge 5) { 4 } else { 2 }
        New-Slot $index 'draw-text' 'text,text_begin'
    }
}

function Get-SlotArgument($Call, $Slot) {
    $arguments = $Call.ArgumentList.Arguments
    foreach ($argument in $arguments) {
        if ($null -ne $argument.NameColon -and
            $argument.NameColon.Name.Identifier.ValueText -cin $Slot.Names) {
            return $argument.Expression
        }
    }
    if ($Slot.Index -ge 0 -and $Slot.Index -lt $arguments.Count -and
        $null -eq $arguments[$Slot.Index].NameColon) {
        return $arguments[$Slot.Index].Expression
    }
    return $null
}

function Get-Label($Expression) {
    if ($null -eq $Expression) { return @{ Kind = 'none'; Text = '<unlabelled>' } }
    if ($Expression -is [Microsoft.CodeAnalysis.CSharp.Syntax.LiteralExpressionSyntax] -and
        $Expression.Token.Value -is [string]) {
        return @{ Kind = 'literal'; Text = $Expression.Token.ValueText }
    }
    $kind = if ($Expression -is [Microsoft.CodeAnalysis.CSharp.Syntax.InterpolatedStringExpressionSyntax]) {
        'interpolated'
    } else { 'dynamic' }
    @{ Kind = $kind; Text = $Expression.ToString() }
}

function Get-MenuPath($Call) {
    $parts = [Collections.Generic.List[string]]::new()
    for ($parent = $Call.Parent; $null -ne $parent; $parent = $parent.Parent) {
        if ($parent -isnot [Microsoft.CodeAnalysis.CSharp.Syntax.IfStatementSyntax] -or
            -not $parent.Statement.Span.Contains($Call.Span)) { continue }
        foreach ($conditionCall in $parent.Condition.DescendantNodesAndSelf()) {
            if ($conditionCall -isnot [Microsoft.CodeAnalysis.CSharp.Syntax.InvocationExpressionSyntax] -or
                (Get-CallName $conditionCall) -cne 'BeginMenu' -or
                $conditionCall.Expression.ToString() -notmatch '(^|\.)ImGui\.BeginMenu$') { continue }
            # Negated conditions usually represent early exits, not open-menu bodies.
            if ($parent.Condition.ToString() -match '!\s*(?:\(\s*)?(?:ImGui|global::ImGuiNET\.ImGui)\.BeginMenu') { continue }
            $label = Get-Label (Get-SlotArgument $conditionCall (New-Slot 0 'label' 'label'))
            $parts.Add($(if ($label.Kind -eq 'literal') { $label.Text.Split('##')[0] } else { '<' + $label.Text + '>' }))
        }
    }
    $parts.Reverse()
    if ((Get-CallName $Call) -cin @('BeginMenu', 'MenuItem')) {
        $label = Get-Label (Get-SlotArgument $Call (New-Slot 0 'label' 'label'))
        $parts.Add($(if ($label.Kind -eq 'literal') { $label.Text.Split('##')[0] } else { '<' + $label.Text + '>' }))
    }
    if ($parts.Count -eq 0) { return '-' }
    $parts -join ' > '
}

function Get-SourceModel([string]$Path, [string]$Text) {
    $options = [Microsoft.CodeAnalysis.CSharp.CSharpParseOptions]::Default.WithPreprocessorSymbols($Define)
    $tree = [Microsoft.CodeAnalysis.CSharp.CSharpSyntaxTree]::ParseText($Text, $options)
    $errors = @($tree.GetDiagnostics() | Where-Object Severity -eq Error)
    if ($errors.Count -gt 0) { throw "C# parse errors in ${Path}: $($errors -join '; ')" }
    $root = $tree.GetRoot()
    $nodes = @($root.DescendantNodes())
    @{
        Path = $Path; Tree = $tree
        Calls = @($nodes | Where-Object { $_ -is [Microsoft.CodeAnalysis.CSharp.Syntax.InvocationExpressionSyntax] })
        Methods = @($nodes | Where-Object {
            $_ -is [Microsoft.CodeAnalysis.CSharp.Syntax.MethodDeclarationSyntax] -or
            $_ -is [Microsoft.CodeAnalysis.CSharp.Syntax.LocalFunctionStatementSyntax]
        })
        ConditionalCount = [regex]::Matches($Text, '(?m)^\s*#(?:if|elif|else)\b').Count
    }
}

function Get-InventoryRows([object[]]$Models) {
    # Infer parameters flowing into UI sinks, including local initializers and
    # forwarding wrappers. Name-only matching is intentionally conservative.
    $definitions = [Collections.Generic.List[object]]::new()
    $byName = [Collections.Generic.Dictionary[string, object]]::new([StringComparer]::Ordinal)
    foreach ($model in $Models) {
        foreach ($method in $model.Methods) {
            $parameters = @($method.ParameterList.Parameters)
            if (@($parameters | Where-Object { $_.Type.ToString() -match '^string(?:\?|\[\])*?$' }).Count -eq 0) { continue }
            $nodes = @($method.DescendantNodes())
            $locals = @{}
            foreach ($node in $nodes) {
                if ($node -is [Microsoft.CodeAnalysis.CSharp.Syntax.VariableDeclaratorSyntax] -and $null -ne $node.Initializer) {
                    $locals[$node.Identifier.ValueText] = $node.Initializer.Value
                }
            }
            $definition = @{
                Parameters = $parameters; Locals = $locals
                Calls = @($nodes | Where-Object { $_ -is [Microsoft.CodeAnalysis.CSharp.Syntax.InvocationExpressionSyntax] })
                Slots = [Collections.Generic.HashSet[int]]::new()
            }
            $definitions.Add($definition)
            $name = $method.Identifier.ValueText
            if (-not $byName.ContainsKey($name)) { $byName[$name] = [Collections.Generic.List[object]]::new() }
            $byName[$name].Add($definition)
        }
    }

    do {
        $changed = $false
        foreach ($definition in $definitions) {
            $identifiers = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
            foreach ($call in $definition.Calls) {
                $slots = @(Get-DirectSlots $call)
                $name = Get-CallName $call
                if ($byName.ContainsKey($name)) {
                    foreach ($target in $byName[$name]) {
                        foreach ($index in $target.Slots) {
                            $slots += New-Slot $index 'wrapper-argument' $target.Parameters[$index].Identifier.ValueText
                        }
                    }
                }
                foreach ($slot in $slots) {
                    $argument = Get-SlotArgument $call $slot
                    if ($null -eq $argument) { continue }
                    foreach ($node in $argument.DescendantNodesAndSelf()) {
                        if ($node -is [Microsoft.CodeAnalysis.CSharp.Syntax.IdentifierNameSyntax]) {
                            [void]$identifiers.Add($node.Identifier.ValueText)
                        }
                    }
                }
            }
            do {
                $expanded = $false
                foreach ($local in $definition.Locals.Keys) {
                    if (-not $identifiers.Contains($local)) { continue }
                    foreach ($node in $definition.Locals[$local].DescendantNodesAndSelf()) {
                        if ($node -is [Microsoft.CodeAnalysis.CSharp.Syntax.IdentifierNameSyntax] -and
                            $identifiers.Add($node.Identifier.ValueText)) { $expanded = $true }
                    }
                }
            } while ($expanded)
            for ($index = 0; $index -lt $definition.Parameters.Count; $index++) {
                $parameter = $definition.Parameters[$index]
                if ($parameter.Type.ToString() -match '^string(?:\?|\[\])*?$' -and
                    $identifiers.Contains($parameter.Identifier.ValueText) -and
                    $definition.Slots.Add($index)) { $changed = $true }
            }
        }
    } while ($changed)

    $rows = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    $callsites = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($model in $Models) {
        foreach ($call in $model.Calls) {
            $slots = @(Get-DirectSlots $call)
            $name = Get-CallName $call
            if ($slots.Count -eq 0 -and $byName.ContainsKey($name)) {
                foreach ($target in $byName[$name]) {
                    foreach ($index in $target.Slots) {
                        $slots += New-Slot $index 'wrapper-argument' $target.Parameters[$index].Identifier.ValueText
                    }
                }
            }
            if ($slots.Count -eq 0) { continue }
            $location = $model.Tree.GetLineSpan($call.Span).StartLinePosition
            $menuPath = Get-MenuPath $call
            foreach ($slot in $slots) {
                $argument = Get-SlotArgument $call $slot
                if ($slot.Index -ge 0 -and $null -eq $argument) { continue }
                # Null shortcuts/previews are not rendered strings.
                if ($null -ne $argument -and $argument.ToString() -ceq 'null') { continue }
                $label = Get-Label $argument
                $fields = @(
                    $model.Path, ('{0:D6}' -f ($location.Line + 1)), $call.Expression.ToString(), $label.Text,
                    ('{0:D4}' -f ($location.Character + 1)), $slot.Role, $label.Kind, $menuPath
                ) | ForEach-Object { ConvertTo-TsvField ([string]$_) }
                [void]$rows.Add(($fields -join "`t"))
                [void]$callsites.Add($model.Path + ':' + $call.SpanStart)
            }
        }
    }
    $sorted = [Collections.Generic.List[string]]::new($rows)
    $sorted.Sort([StringComparer]::Ordinal)
    @{ Rows = $sorted; Callsites = $callsites.Count; WrapperDefinitions = @($definitions | Where-Object { $_.Slots.Count -gt 0 }).Count }
}

function Invoke-SelfTest {
    $source = @'
class Fixture {
    void Wrapper(string caption) { var text = $"wrapped {caption}"; Forward(text); }
    void Forward(string label) => ImGui.Text(label);
    void Draw() {
        // ImGui.Button("comment must not appear");
        var fake = "ImGui.Button(\"string must not appear\")";
        if (ImGui.BeginMenu("File")) {
            if (ImGui.BeginMenu("Export##id")) {
                ImGui.MenuItem("Mesh", "Ctrl+E");
                ImGui.TextColored(
                    color, $"Value {value:N2}");
                Wrapper("literal wrapper");
                ImGui.TreeNodeEx("##node", flags, "Visible node");
            } else { ImGui.Button("not in Export"); }
        }
        ImGui.Button(label: "named");
        ImGui.MenuItem("Enabled item", enabled);
        ImGui.InputTextWithHint("##search", "type\there\nnow", ref text, 20);
        drawList.AddText(font, 12, pos, color, "drawn");
        ImGui.Combo("Choice", ref choice, new[] { "One", "Two" }, 2);
        ImGui.Button(@"verbatim\label"); ImGui.Button("same line");
        ImGui.Text("""raw label""");
        ImGui.Image(texture, size);
    }
}
'@
    $result = Get-InventoryRows @((Get-SourceModel 'fixture.cs' $source))
    $text = $result.Rows -join "`n"
    $checks = [ordered]@{
        'comments and string contents ignored' = $text -notmatch 'must not appear'
        'nested menu path and hidden suffix' = $text.Contains("File > Export > Mesh")
        'else is not child of Export' = @($result.Rows | Where-Object { $_ -match 'not in Export' -and $_.EndsWith("`tFile") }).Count -eq 1
        'multiline interpolated colored text' = $text.Contains('Value {value:N2}') -and $text.Contains("`tinterpolated`t")
        'transitive wrapper and local variable inference' = $text.Contains("Wrapper`tliteral wrapper`t")
        'named argument' = $text.Contains("ImGui.Button`tnamed`t")
        'two-argument menu enablement is not text' = $text -notmatch 'ImGui.MenuItem\tenabled\t'
        'TreeNodeEx display-text overload' = $text.Contains("ImGui.TreeNodeEx`tVisible node`t")
        'hint escaping' = $text.Contains('type\there\nnow')
        'font draw-list overload' = $text.Contains("drawList.AddText`tdrawn`t")
        'combo option source expression' = $text.Contains('new[] { "One", "Two" }')
        'verbatim literal' = $text.Contains('verbatim\\label')
        'raw literal' = $text.Contains("ImGui.Text`traw label`t")
        'unlabelled control' = $text.Contains("ImGui.Image`t<unlabelled>`t")
        'same-line distinct callsites' = @($result.Rows | Where-Object { $_ -match 'ImGui.Button\t(?:same line|verbatim)' }).Count -eq 2
        'repeatable ordinal ordering' = ($text -ceq ((Get-InventoryRows @((Get-SourceModel 'fixture.cs' $source))).Rows -join "`n"))
    }
    foreach ($check in $checks.GetEnumerator()) {
        if (-not $check.Value) { throw "SELF_TEST_FAIL: $($check.Key)" }
    }
    Write-Output "SELF_TEST_PASS checks=$($checks.Count) rows=$($result.Rows.Count)"
}

function Get-ViewerFiles([string]$Root) {
    $pending = [Collections.Generic.Stack[string]]::new()
    $pending.Push($Root)
    while ($pending.Count -gt 0) {
        foreach ($entry in Get-ChildItem -LiteralPath $pending.Pop() -Force) {
            if ($entry.PSIsContainer) {
                if ($entry.Name -match 'Defunct|^(bin|obj)$') { continue }
                if ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Refusing directory link: $($entry.FullName)" }
                $pending.Push($entry.FullName)
            } elseif ($entry.Extension -ieq '.cs') { $entry.FullName }
        }
    }
}

if ($SelfTest) { Invoke-SelfTest; return }
if ($Force -and $Verify) { throw '-Force and -Verify cannot be combined.' }
$repo = [IO.Path]::GetFullPath($RepositoryRoot)
$viewer = [IO.Path]::GetFullPath((Join-Path $repo $ViewerPath))
$output = [IO.Path]::GetFullPath((Join-Path $repo $OutputPath))
$relativeOutput = [IO.Path]::GetRelativePath($repo, $output)
if ([IO.Path]::IsPathRooted($relativeOutput) -or $relativeOutput -match '^\.\.([\\/]|$)') {
    throw 'Output must be inside RepositoryRoot.'
}
if ([IO.Path]::GetExtension($output) -ine '.txt') { throw 'Output must be a .txt inventory, not a source file.' }
if (-not $Verify -and -not $Force -and [IO.File]::Exists($output)) { throw 'Output already exists; use -Verify, a new path, or explicit -Force.' }
$paths = [Collections.Generic.List[string]]::new()
foreach ($path in Get-ViewerFiles $viewer) { $paths.Add($path) }
$paths.Sort([StringComparer]::Ordinal)
if ($paths.Count -eq 0) { throw 'No viewer C# files found.' }
$models = [Collections.Generic.List[object]]::new()
$hashes = [Collections.Generic.Dictionary[string, string]]::new([StringComparer]::Ordinal)
$manifest = [Collections.Generic.List[string]]::new()
$conditionalCount = 0
foreach ($path in $paths) {
    $relative = [IO.Path]::GetRelativePath($repo, $path).Replace('\', '/')
    $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    $hashes[$path] = $hash
    $manifest.Add($relative + "`t" + $hash)
    $model = Get-SourceModel $relative ([IO.File]::ReadAllText($path))
    $models.Add($model)
    $conditionalCount += $model.ConditionalCount
}
$result = Get-InventoryRows $models.ToArray()
$after = [Collections.Generic.List[string]]::new()
foreach ($path in Get-ViewerFiles $viewer) { $after.Add($path) }
$after.Sort([StringComparer]::Ordinal)
if (($after -join "`n") -cne ($paths -join "`n")) { throw 'Viewer file set changed during inventory; retry before recording baseline.' }
foreach ($path in $paths) {
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -cne $hashes[$path]) {
        throw "Source changed during inventory: $path"
    }
}
$digest = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes(($manifest -join "`n") + "`n")))
$header = @(
    '# UI string inventory v1; static source evidence, NOT runtime completeness',
    ('# viewer=' + [IO.Path]::GetRelativePath($repo, $viewer).Replace('\', '/')),
    '# excludes=directory names containing Defunct, bin, obj (case-insensitive)',
    "# files=$($paths.Count); rows=$($result.Rows.Count); callsites=$($result.Callsites); wrapper_definitions=$($result.WrapperDefinitions)",
    "# source_sha256=$digest",
    "# conditional_directives=$conditionalCount; defines=$(@($Define | Sort-Object -CaseSensitive) -join ',')",
    '# menu_path=lexical positive BeginMenu if-branch ancestry; helpers may have unknown ancestors (-)',
    '# label=decoded literal or unevaluated source expression; ## IDs retained; TSV escapes=\\,\t,\r,\n',
    '# wrapper-argument=name/parameter heuristic, not semantic binding; options are not expanded',
    "file`tline`tcontrol`tlabel`tcolumn`trole`tkind`tmenu_path"
)
$content = ($header -join "`n") + "`n" + ($result.Rows -join "`n") + "`n"
$bytes = [Text.UTF8Encoding]::new($false).GetBytes($content)
if ($Verify) {
    if (-not [IO.File]::Exists($output) -or
        [Convert]::ToBase64String([IO.File]::ReadAllBytes($output)) -cne [Convert]::ToBase64String($bytes)) {
        throw 'Inventory differs from output (labels, locations, or source fingerprint changed).'
    }
} else {
    [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($output))
    [IO.File]::WriteAllBytes($output, $bytes)
}
$verb = if ($Verify) { 'VERIFIED' } else { 'WRITTEN' }
Write-Output "UI_INVENTORY_$verb files=$($paths.Count) rows=$($result.Rows.Count) callsites=$($result.Callsites) wrapper_definitions=$($result.WrapperDefinitions) conditional_directives=$conditionalCount"
Write-Output "SOURCE_SHA256=$digest"
Write-Output ('OUTPUT=' + $relativeOutput.Replace('\', '/'))
