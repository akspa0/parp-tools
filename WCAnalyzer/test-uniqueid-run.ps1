# Test script for running UniqueIdAnalyzer with sample data

# Build the solution
dotnet build WCAnalyzer.sln

# Create test directories
$resultsDir = "test-results"
$outputDir = "test-output"

# Create directories if they don't exist
if (-not (Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir
}

if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir
}

# Run the UniqueIdAnalyzer command
dotnet run --project WCAnalyzer.CLI/WCAnalyzer.CLI.csproj -- uniqueid --results-directory $resultsDir --output $outputDir --verbose 