These files tell WoW which Shaders to use on different surfaces. The files are hardcoded into the client. As of 3.0.9, WoW loads:
*MapObj.wfx
**Particle (''//Particle shaders - move to another file.'')
**MapObjDiffuse
**MapObjOpaque
**MapObjSpecular
**MapObjMetal
**MapObjEnv
**MapObjEnvMetal
*MapObjU.wfx
**MapObjUDiffuse
**MapObjUOpaque
**MapObjUSpecular
**MapObjUMetal
**MapObjUEnv
**MapObjUEnvMetal
*Model2.wfx
**Projected_ModMod
**Projected_ModAdd
*ShadowMap.wfx
**ShadowMapRender
**ShadowMapRenderSL

'''Bold''' characters are needed for parsing the file right. Non-bold functions are optional.

The shaders referenced in the RenderState*()s are references to the [[BLS]] files.

 '''Effect(name, unk)'''	// this name is referenced from the app (hardcoded)
 '''{'''
 	'''FixedFunc()'''
 	'''{'''
 		'''Pass(type, passCount)'''                     // <=1 asserted
 		'''{'''
 			ColorOp0(mode);
 			ColorOp1(mode);
 			AlphaOp0(mode);
 			AlphaOp1(mode);
 			RenderStateARGB(shader, color, color); // see list below
 			RenderStateF(shader, float, float); // float
 			RenderStateI(shader, integer, integer); // see list below
 		'''}'''
 '''	}'''
 	'''Shader()'''
 	'''{'''
 		'''Pass(type, passCount)'''                    // ==1 asserted
 		'''{'''
 			VertexShader(mode);
 			PixelShader(mode);
 			Specular(mode);
 			RenderStateARGB(shader, color, color); // see list below
 			RenderStateF(shader, float, float); // float
 			RenderStateI(shader, integer, integer); // see list below
 		'''}'''
 '''	}'''
 '''}'''
==Pass: type==
*Default
*Opaque
*AlphaKey
*Alpha
*Add
*Mod
*Mod2x
*ModAdd
*InvSrcAlphaAdd
*InvSrcAlphaOpaque
*SrcAlphaOpaque
*NoAlphaAdd
*ConstantAlpha

==*****Op*: mode==
*Mod
*Mod2x
*Add
*PassThru
*Decal
*Fade

==RenderStateARGB: color==
*white
*black
*every value getting a good result with strtoul(string, 0, 0x10)

==RenderStateI: integer==
*true
*false
*TexGen_Disable
*TexGen_Object
*TexGen_World
*TexGen_View
*TexGen_ViewReflection
*TexGen_ViewNormal
*TexGen_SphereMap
*TS_PassThru
*TS_Affine
*TS_Proj

==RenderState: type==
*MatDiffuse
*MatEmissive
*MatSpecular
*MatSpecularExp
*NormalizeNormals
*SceneAmbient
*DepthTest
*DepthFunc
*DepthWrite
*ColorWrite
*Culling
*ClipPlaneMask
*Lighting
*TexLodBiasZ // Z=0-7
*TexGenZ // Z=0-7
*TextureShaderZ // Z=0-7
*PointScale
*PointScaleAttenuation
*PointScaleMin
*PointScaleMax
*PointSprite
*ConstBlendAlpha
*Unknown
[[Category:Format]]
