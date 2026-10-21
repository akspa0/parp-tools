# MainMenu source-only inventory
```text
# UI string inventory v1; static source evidence, NOT runtime completeness
# viewer=wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu
# excludes=directory names containing Defunct, bin, obj (case-insensitive)
# files=4; rows=64; callsites=59; wrapper_definitions=1
# source_sha256=8762EBE87D3CDCCF5662F91ADDD09F3CDBEA531B7850165E8DF8280144BF3ADF
# conditional_directives=0; defines=
# menu_path=lexical positive BeginMenu if-branch ancestry; helpers may have unknown ancestors (-)
# label=decoded literal or unevaluated source expression; ## IDs retained; TSV escapes=\\,\t,\r,\n
# wrapper-argument=name/parameter heuristic, not semantic binding; options are not expanded
file	line	control	label	column	role	kind	menu_path
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.Content.cs	000024	ImGui.TextWrapped	Convert maps using the existing converter service and its output manifest.	0013	text	literal	-
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.Content.cs	000033	ImGui.TextWrapped	Convert WMO files using the existing converter service.	0013	text	literal	-
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.Content.cs	000052	ImGui.TextDisabled	Uses the existing DAT exporter and its output/manifest status; runtime confirmation is operator-owned.	0013	text	literal	-
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.Content.cs	000097	ImGui.TextWrapped	$"Remove saved base '{name}'?"	0017	text	interpolated	-
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.Content.cs	000098	ImGui.TextWrapped	_pendingForgetKnownGoodClientPath	0084	text	dynamic	-
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.Content.cs	000122	ImGui.TextUnformatted	ViewerProductName	0009	text	dynamic	-
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.Content.cs	000123	ImGui.TextDisabled	$"Version {GetViewerDisplayVersion()}"	0009	text	interpolated	-
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.Content.cs	000125	ImGui.TextWrapped	World/model viewer and debugging surface for WoW Alpha, Wrath, and early Cataclysm data.	0009	text	literal	-
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.Content.cs	000127	ImGui.TextWrapped	Author: github.com/akspa0/parp-tools	0009	text	literal	-
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.Content.cs	000128	ImGui.TextWrapped	Discord: discord.gg/6YdUksuKuU	0009	text	literal	-
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.Content.cs	000130	ImGui.TextUnformatted	In memory of Hayven Games	0009	text	literal	-
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.Content.cs	000131	ImGui.TextWrapped	An inspiration for this project and a friend, whose short films explored World of Warcraft's secrets and little-known details through game footage.	0009	text	literal	-
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.Content.cs	000134	ImGui.TextWrapped	Thanks to...	0009	text	literal	-
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.Content.cs	000135	ImGui.TextWrapped	Marlamin, schlumpf, Dovah, Pirate the Explorer, fean, implave, IS4, Mjollna, Adspartan (Noggit), and Skarn (Noggit-Red).	0009	text	literal	-
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.Content.cs	000136	ImGui.TextWrapped	Special thanks to Kruithne for wow.export, whose tools and format research have been invaluable to the community.	0009	text	literal	-
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.Content.cs	000137	ImGui.TextWrapped	Without the WoW Exploration community, this project would not exist. Everyone named here contributed inspiration to this project in some way.	0009	text	literal	-
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.Content.cs	000138	ImGui.TextDisabled	This tooling is about restoration, not touching up or polishing what we recover - it is the instrument for restoring what already exists. Noggit and Noggit-Red remain the preferred editors for fine-tuning the results this library and tooling produce.	0009	text	literal	-
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.Content.cs	000140	ImGui.TextWrapped	Special thanks to WoWdev.wiki, Exploration Reboot, The Alpha Project, and everyone in the Pre-Alpha Restoration Project discord!	0009	text	literal	-
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.ViewerKeyBindings.cs	000015	ImGui.TextUnformatted	Keyboard Shortcuts	0013	text	literal	-
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.ViewerKeyBindings.cs	000016	ImGui.TextDisabled	$"Active page: {activeContext}"	0013	text	interpolated	-
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.ViewerKeyBindings.cs	000019	ImGui.BeginChild	##KeyboardShortcutsBody	0017	id	literal	-
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.ViewerKeyBindings.cs	000023	DrawKeyboardBindingGroup	Global	0017	wrapper-argument	literal	-
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.ViewerKeyBindings.cs	000037	DrawKeyboardBindingGroup	title	0021	wrapper-argument	dynamic	-
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.ViewerKeyBindings.cs	000045	ImGui.TextUnformatted	title	0009	text	dynamic	-
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.ViewerKeyBindings.cs	000050	ImGui.TextDisabled	binding.Gesture	0013	text	dynamic	-
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.ViewerKeyBindings.cs	000052	ImGui.TextWrapped	binding.Description	0013	text	dynamic	-
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.ViewerKeyBindings.cs	000056	ImGui.TextDisabled	No page-specific shortcuts are assigned here.	0013	text	literal	-
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000074	ImGui.BeginMenu	File	0017	label	literal	File
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000076	ImGui.MenuItem	Open File...	0021	label	literal	File > Open File...
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000079	ImGui.MenuItem	Open Alpha WDT (loose map)...	0021	label	literal	File > Open Alpha WDT (loose map)...
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000082	ImGui.MenuItem	Open Loose PM4 / PD4 File...	0021	label	literal	File > Open Loose PM4 / PD4 File...
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000087	ImGui.MenuItem	Open Game Folder (MPQ)...	0021	label	literal	File > Open Game Folder (MPQ)...
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000093	ImGui.MenuItem	Open CASC Install (local)...	0021	label	literal	File > Open CASC Install (local)...
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000097	ImGui.SetTooltip	Pick a Battle.net install folder, then the game version to load. Reads only what is on disk.	0021	text	literal	File
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000099	ImGui.MenuItem	Open CASC Install (local + CDN fill)...	0021	label	literal	File > Open CASC Install (local + CDN fill)...
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000103	ImGui.SetTooltip	Pick a Battle.net install folder, then the game version to load. Files the build lists but the install does not have on disk are downloaded from Blizzard's CDN for the same build.	0021	text	literal	File
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000105	ImGui.MenuItem	Open DAT Terrain Folder...	0021	label	literal	File > Open DAT Terrain Folder...
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000107	ImGui.BeginMenu	Open Saved Game Folder	0021	label	literal	File > Open Saved Game Folder
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000111	ImGui.MenuItem	$"{knownClient.Name}##open_saved_{knownClient.Path}"	0029	label	interpolated	File > Open Saved Game Folder > <$"{knownClient.Name}##open_saved_{knownClient.Path}">
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000115	ImGui.SetTooltip	ClientDialogsService.BuildKnownGoodClientTooltip(knownClient)	0029	text	dynamic	File > Open Saved Game Folder
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000121	ImGui.MenuItem		0021	shortcut	literal	File > Attach Loose Map Folder...
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000121	ImGui.MenuItem	Attach Loose Map Folder...	0021	label	literal	File > Attach Loose Map Folder...
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000124	ImGui.BeginMenu	Load Loose Map Folder Against Saved Base	0021	label	literal	File > Load Loose Map Folder Against Saved Base
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000128	ImGui.MenuItem	$"{knownClient.Name}##attach_saved_{knownClient.Path}"	0029	label	interpolated	File > Load Loose Map Folder Against Saved Base > <$"{knownClient.Name}##attach_saved_{knownClient.Path}">
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000132	ImGui.SetTooltip	ClientDialogsService.BuildKnownGoodClientTooltip(knownClient)	0029	text	dynamic	File > Load Loose Map Folder Against Saved Base
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000140	ImGui.MenuItem		0021	shortcut	literal	File > Save Current Game Folder As Known-Good Base
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000140	ImGui.MenuItem	Save Current Game Folder As Known-Good Base	0021	label	literal	File > Save Current Game Folder As Known-Good Base
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000143	ImGui.BeginMenu	Forget Known-Good Base	0021	label	literal	File > Forget Known-Good Base
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000147	ImGui.MenuItem	$"{knownClient.Name}##forget_saved_{knownClient.Path}"	0029	label	interpolated	File > Forget Known-Good Base > <$"{knownClient.Name}##forget_saved_{knownClient.Path}">
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000151	ImGui.SetTooltip	ClientDialogsService.BuildKnownGoodClientTooltip(knownClient)	0029	text	dynamic	File > Forget Known-Good Base
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000159	ImGui.MenuItem	Settings...	0021	label	literal	File > Settings...
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000164	ImGui.MenuItem	Quit	0021	label	literal	File > Quit
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000169	ImGui.BeginMenu	View	0013	label	literal	View
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000171	ImGui.MenuItem	Reset Camera	0017	label	literal	View > Reset Camera
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000172	ImGui.MenuItem	Hide UI Chrome	0017	label	literal	View > Hide UI Chrome
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000172	ImGui.MenuItem	Tab	0017	shortcut	literal	View > Hide UI Chrome
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000173	ImGui.MenuItem		0013	shortcut	literal	View > Left Sidebar
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000173	ImGui.MenuItem	Left Sidebar	0013	label	literal	View > Left Sidebar
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000174	ImGui.MenuItem	I	0013	shortcut	literal	View > Right Sidebar
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000174	ImGui.MenuItem	Right Sidebar	0013	label	literal	View > Right Sidebar
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000175	ImGui.MenuItem	Reset Shell Layout	0017	label	literal	View > Reset Shell Layout
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000178	ImGui.BeginMenu	Help	0017	label	literal	Help
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000180	ImGui.MenuItem	Keyboard Shortcuts	0021	label	literal	Help > Keyboard Shortcuts
wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs	000183	ImGui.MenuItem	About	0021	label	literal	Help > About
```
