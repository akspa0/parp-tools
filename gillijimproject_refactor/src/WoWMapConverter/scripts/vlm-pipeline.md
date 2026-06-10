---
description: End-to-end pipeline for creating a VLM training dataset from WoW maps.
---

This workflow automates the process of exporting raw map data, curating a diverse subset for training, and formatting it for Unsloth/Qwen3-VL fine-tuning.

# 1. Export Raw Data (C#)
First, export the full map data using the C# tool. This extracts terrain, textures, and objects from the client MPQs.
*Replace paths with your actual client and output locations.*

```powershell
dotnet run --project src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -- vlm-export --client "C:\WoW335\3.X_Retail_Windows_enUS_3.3.5.12340\World of Warcraft\" --map "Azeroth" --out "j:\vlm_output\full_azeroth"
```

# 2. Curate Dataset (Python)
Select a balanced subset of tiles to train on. This script analyzes the exported JSONs and picks a mix of simple, complex, and feature-rich tiles (e.g., water, dense objects) to prevent dataset imbalance.
*Adjust `--count` to your desired dataset size (e.g., 500).*

```powershell
python src/WoWMapConverter/scripts/vlm_curate.py "j:\vlm_output\full_azeroth" "j:\vlm_output\curated_azeroth" --count 500
```

# 3. Format for Unsloth (Python)
Convert the curated dataset into the JSONL format required by Unsloth for Qwen3-VL fine-tuning. This embeds the instruction prompts and terrain summaries.

```powershell
python src/WoWMapConverter/scripts/vlm_to_unsloth.py "j:\vlm_output\curated_azeroth" -o "j:\vlm_output\curated_azeroth\train.jsonl"
```

# 4. Result
You now have `j:\vlm_output\curated_azeroth\train.jsonl` ready for upload to Colab or local training!
