# ChunkVault Parser Script
# This script helps parse documentation files and extract chunks into the ChunkVault

param(
    [Parameter(Mandatory=$true)]
    [string]$DocumentationFile,
    
    [Parameter(Mandatory=$false)]
    [string]$OutputDir = ".context/chunkvault/chunks",
    
    [Parameter(Mandatory=$false)]
    [string]$ChunkCategory = ""
)

# Ensure the output directory exists
$categoryDir = Join-Path $OutputDir $ChunkCategory
if (!(Test-Path $categoryDir)) {
    New-Item -ItemType Directory -Path $categoryDir -Force
}

# Read the documentation file
$content = Get-Content $DocumentationFile -Raw

# Extract file format from filename
$fileFormat = [System.IO.Path]::GetFileNameWithoutExtension($DocumentationFile)

# Regular expression to find chunk headers
# This is a simplified pattern that looks for chunk names (e.g., "## MVER chunk")
$chunkPattern = "(?m)^#+\s+([A-Z0-9]{4})\s+chunk"

# Find all chunks in the documentation
$matches = [regex]::Matches($content, $chunkPattern)

Write-Host "Found $($matches.Count) potential chunks in $DocumentationFile"

# Process each chunk
$chunkCounter = 0
foreach ($match in $matches) {
    $chunkName = $match.Groups[1].Value
    $chunkCounter++
    
    # Find the chunk description
    $startIndex = $match.Index
    $nextChunkIndex = $content.IndexOf("chunk", $startIndex + $match.Length)
    
    if ($nextChunkIndex -eq -1) {
        $nextChunkIndex = $content.Length
    }
    
    $chunkContent = $content.Substring($startIndex, $nextChunkIndex - $startIndex)
    
    # Extract struct definition if present
    $structDef = ""
    if ($chunkContent -match "```(?:[a-z]*)\s*(struct\s+[^`]+)```") {
        $structDef = $matches[1]
    }
    
    # Create unique ID for the chunk
    [string]$formattedCounter = $chunkCounter.ToString("000")
    [string]$chunkId = "C$formattedCounter"
    
    # Create output file
    $outputFile = Join-Path $categoryDir "$chunkId`_$chunkName.md"
    
    Write-Host "Processing $chunkName as $chunkId"
    
    # Create the header section
    $header = "# $chunkId`: $chunkName`n`n"
    $header += "## Type`n$fileFormat Chunk`n`n"
    $header += "## Source`n$([System.IO.Path]::GetFileName($DocumentationFile))`n`n"
    $header += "## Description`n[Extract from documentation]`n`n"
    
    # Create the structure section
    $structure = "## Structure`n```csharp`n$structDef`n````n`n"
    
    # Create the properties section
    $properties = "## Properties`n| Name | Type | Description |`n|------|------|-------------|`n| [Property] | [Type] | [Description] |`n`n"
    
    # Create the dependencies section
    $dependencies = "## Dependencies`n[Dependencies from other chunks]`n`n"
    
    # Create the notes section
    $notes = "## Implementation Notes`n- [Important notes from documentation]`n`n"
    
    # Create the example section
    $example = "## Implementation Example`n```csharp`npublic class $chunkName`n{`n    // Properties go here`n}`n````n`n"
    
    # Create the usage section
    $usage = "## Usage Context`n[How this chunk is used in the file format]"
    
    # Combine all sections
    $output = $header + $structure + $properties + $dependencies + $notes + $example + $usage
    
    # Write to output file
    Set-Content $outputFile $output
    
    Write-Host "Created chunk file: $outputFile"
}

# Update the index file with the new chunks
$indexFile = ".context/chunkvault/index.md"
$indexContent = Get-Content $indexFile

# Extract the current registry section
$registrySection = [regex]::Match($indexContent, "(?s)## Chunk Registry\s*\|\s*ID\s*\|\s*Name.*?\n\n").Value

# Update counts in the Progress Summary section
$indexContent = $indexContent -replace "Total Chunks: \[Count\]", "Total Chunks: $chunkCounter"
$indexContent = $indexContent -replace "Pending: \[Count\]", "Pending: $chunkCounter"

# Write the updated index
Set-Content $indexFile $indexContent

Write-Host "Updated index file with $chunkCounter chunks"
Write-Host "Done processing $DocumentationFile" 