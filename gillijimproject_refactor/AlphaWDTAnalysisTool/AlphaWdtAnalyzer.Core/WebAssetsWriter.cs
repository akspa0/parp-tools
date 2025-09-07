using System.IO;
using System.Text;

namespace AlphaWdtAnalyzer.Core;

public static class WebAssetsWriter
{
    public static void Write(string outDir)
    {
        var webDir = Path.Combine(outDir, "web");
        Directory.CreateDirectory(webDir);

        var indexHtml = @"<!doctype html>
<html lang=""en"">
<head>
  <meta charset=""utf-8"" />
  <meta name=""viewport"" content=""width=device-width, initial-scale=1"" />
  <title>Alpha WDT Analyzer</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 1rem 2rem; }
    .counts span { display: inline-block; margin-right: 1rem; }
    .section { margin-top: 1.5rem; }
    ul { max-height: 300px; overflow: auto; border: 1px solid #ddd; padding: .5rem; }
    code { background: #f5f5f5; padding: .1rem .25rem; }
  </style>
</head>
<body>
  <h1 id=""title"">Alpha WDT Analyzer</h1>
  <div class=""counts"" id=""counts""></div>

  <div class=""section"">
    <h2>Tiles</h2>
    <ul id=""tiles""></ul>
  </div>

  <div class=""section"">
    <h2>Assets - WMO</h2>
    <ul id=""wmo""></ul>
  </div>
  <div class=""section"">
    <h2>Assets - M2/MDX</h2>
    <ul id=""m2""></ul>
  </div>
  <div class=""section"">
    <h2>Assets - BLP</h2>
    <ul id=""blp""></ul>
  </div>
  <div class=""section"">
    <h2>Missing (vs listfile)</h2>
    <h3>WMO</h3>
    <ul id=""missWmo""></ul>
    <h3>M2</h3>
    <ul id=""missM2""></ul>
    <h3>BLP</h3>
    <ul id=""missBlp""></ul>
  </div>

  <script src=""app.js""></script>
</body>
</html>";

        var appJs = @"async function main(){
  const res = await fetch('../index.json');
  const idx = await res.json();
  document.getElementById('title').textContent = `Alpha WDT Analyzer - ${idx.MapName}`;
  const counts = document.getElementById('counts');
  counts.innerHTML = `
    <span><b>WMO</b>: ${idx.WmoAssets.length}</span>
    <span><b>M2</b>: ${idx.M2Assets.length}</span>
    <span><b>BLP</b>: ${idx.BlpAssets.length}</span>
    <span><b>Tiles</b>: ${idx.Tiles.length}</span>`;

  function fillList(id, arr){
    const ul = document.getElementById(id);
    ul.innerHTML='';
    arr.forEach(item=>{
      const li=document.createElement('li');
      li.textContent=item.AssetPath || item;
      ul.appendChild(li);
    });
  }

  fillList('tiles', idx.Tiles.map(t=>`${t.X}_${t.Y} - ${t.AdtPath}`));
  fillList('wmo', idx.WmoAssets);
  fillList('m2', idx.M2Assets);
  fillList('blp', idx.BlpAssets);
  fillList('missWmo', idx.MissingWmo);
  fillList('missM2', idx.MissingM2);
  fillList('missBlp', idx.MissingBlp);
}
main().catch(e=>{
  document.body.insertAdjacentHTML('beforeend', `<pre style=""color:red"">${e}</pre>`);
});";

        File.WriteAllText(Path.Combine(webDir, "index.html"), indexHtml, Encoding.UTF8);
        File.WriteAllText(Path.Combine(webDir, "app.js"), appJs, Encoding.UTF8);
    }
}
