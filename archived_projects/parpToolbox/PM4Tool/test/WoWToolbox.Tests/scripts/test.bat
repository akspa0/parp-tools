@echo off
echo Running WoWToolbox.Tests tests...
dotnet test ../WoWToolbox.Tests.csproj
if %errorlevel% neq 0 (
    echo Tests failed!
    exit /b %errorlevel%
)
echo Tests successful. 