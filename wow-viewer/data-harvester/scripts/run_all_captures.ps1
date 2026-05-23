# Run each build in a separate PowerShell window.
# Each instance reads a different staged client and writes to its own capture dir.
# The viewer is a GUI app (OpenGL) so it needs a display — can't run truly headless.

$MDXVIEWER = "I:\parp\parp-tools\gillijimproject_refactor\src\MdxViewer\bin\Debug\net10.0-windows\ParpToolsWoWViewer.exe"
$CLIENT = "I:\parp\parp-tools\output\tmp\wowarchive-clients"
$CAPTURE = "I:\parp\parp-tools\output\tmp\mdxviewer_validation_smoke"
$LISTFILE = "I:\parp\parp-tools\gillijimproject_refactor\test_data\community-listfile-withcapitals.csv"
$RES = 512

# Build 0_5_3_3368 (ALREADY RUNNING — skip if active)
# & $MDXVIEWER --game-path "$CLIENT\0_5_3_3368\World of Warcraft" --build 0.5.3.3368 --listfile $LISTFILE --world "World\Maps\Azeroth\Azeroth.wdt" --validation-dataset-root "$CAPTURE\0_5_3_3368" --validation-output "$CAPTURE\0_5_3_3368" --validation-resolution $RES --force-validation-regeneration --exit-after-validation

# Build 0_5_5_3494
& $MDXVIEWER --game-path "$CLIENT\0_5_5_3494\World of Warcraft" --build 0.5.5.3494 --listfile $LISTFILE --world "World\Maps\Azeroth\Azeroth.wdt" --validation-dataset-root "$CAPTURE\0_5_5_3494" --validation-output "$CAPTURE\0_5_5_3494" --validation-resolution $RES --force-validation-regeneration --exit-after-validation

# Build 0_7_0_3694
& $MDXVIEWER --game-path "$CLIENT\0_7_0_3694\World of Warcraft" --build 0.7.0.3694 --listfile $LISTFILE --world "World\Maps\Azeroth\Azeroth.wdt" --validation-dataset-root "$CAPTURE\0_7_0_3694" --validation-output "$CAPTURE\0_7_0_3694" --validation-resolution $RES --force-validation-regeneration --exit-after-validation

# Build 3_0_1_8303 (has multiple maps)
& $MDXVIEWER --game-path "$CLIENT\3_0_1_8303\World of Warcraft" --build 3.0.1.8303 --listfile $LISTFILE --world "World\Maps\Azeroth\Azeroth.wdt" --validation-dataset-root "$CAPTURE\3_0_1_8303" --validation-output "$CAPTURE\3_0_1_8303" --validation-resolution $RES --force-validation-regeneration --exit-after-validation

# Build 3_3_5_12340
& $MDXVIEWER --game-path "$CLIENT\3_3_5_12340\World of Warcraft" --build 3.3.5.12340 --listfile $LISTFILE --world "World\Maps\Azeroth\Azeroth.wdt" --validation-dataset-root "$CAPTURE\3_3_5_12340" --validation-output "$CAPTURE\3_3_5_12340" --validation-resolution $RES --force-validation-regeneration --exit-after-validation

# Build 4_0_0_11927
& $MDXVIEWER --game-path "$CLIENT\4_0_0_11927\World of Warcraft" --build 4.0.0.11927 --listfile $LISTFILE --world "World\Maps\Azeroth\Azeroth.wdt" --validation-dataset-root "$CAPTURE\4_0_0_11927" --validation-output "$CAPTURE\4_0_0_11927" --validation-resolution $RES --force-validation-regeneration --exit-after-validation
