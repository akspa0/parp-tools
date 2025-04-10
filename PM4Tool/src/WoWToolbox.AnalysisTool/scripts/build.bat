@echo off
echo Building WoWToolbox.AnalysisTool project...
dotnet build ../WoWToolbox.AnalysisTool.csproj
if %errorlevel% neq 0 (
    echo Build failed!
    exit /b %errorlevel%
)
echo Build successful. 