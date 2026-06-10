@echo off
echo Building WoWToolbox.Tests project...
dotnet build ../WoWToolbox.Tests.csproj
if %errorlevel% neq 0 (
    echo Build failed!
    exit /b %errorlevel%
)
echo Build successful. 