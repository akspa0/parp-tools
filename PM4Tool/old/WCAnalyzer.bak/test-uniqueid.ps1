# Test script for UniqueIdAnalyzer

# Build the solution
dotnet build WCAnalyzer.sln

# Run the UniqueIdAnalyzer command
dotnet run --project WCAnalyzer.CLI/WCAnalyzer.CLI.csproj -- uniqueid --help 