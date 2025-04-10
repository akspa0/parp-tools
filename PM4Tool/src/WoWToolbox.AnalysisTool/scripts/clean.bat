@echo off
echo Cleaning WoWToolbox.AnalysisTool project...
dotnet clean ../WoWToolbox.AnalysisTool.csproj
if %errorlevel% neq 0 (
    echo Clean failed!
    exit /b %errorlevel%
)
echo Clean successful. 