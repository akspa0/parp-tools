@echo off
echo Cleaning WoWToolbox.Tests project...
dotnet clean ../WoWToolbox.Tests.csproj
if %errorlevel% neq 0 (
    echo Clean failed!
    exit /b %errorlevel%
)
echo Clean successful. 