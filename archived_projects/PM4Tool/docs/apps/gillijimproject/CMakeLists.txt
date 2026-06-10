cmake_minimum_required (VERSION 2.6)
project (gillijimproject)

set ( gp_sources
  utilities/Utilities.cpp

  wowfiles/alpha/AdtAlpha.cpp
  wowfiles/alpha/WdtAlpha.cpp
  wowfiles/alpha/McnkAlpha.cpp  
  wowfiles/alpha/McnrAlpha.cpp  
  wowfiles/alpha/McvtAlpha.cpp  
  wowfiles/alpha/Mdnm.cpp  
  wowfiles/alpha/Monm.cpp  
  wowfiles/alpha/MphdAlpha.cpp  
  wowfiles/alpha/MainAlpha.cpp  
  
  wowfiles/cataclysm/AdtCata.cpp  
  wowfiles/cataclysm/AdtCataTerrain.cpp  
  wowfiles/cataclysm/AdtCataTextures.cpp  
  wowfiles/cataclysm/AdtCataObjects.cpp  
  wowfiles/cataclysm/McnkCataTerrain.cpp  
  wowfiles/cataclysm/McnkCataObjects.cpp  
  wowfiles/cataclysm/McnkCataTextures.cpp
  wowfiles/cataclysm/McnrCata.cpp  
  
  wowfiles/lichking/AdtLk.cpp
  wowfiles/lichking/McnkLk.cpp 
  wowfiles/lichking/McnrLk.cpp
  
  wowfiles/Chunk.cpp  
  wowfiles/Main.cpp  
  wowfiles/Mcal.cpp  
  wowfiles/Mcin.cpp 
  wowfiles/Mcnk.cpp 
  wowfiles/Mcrf.cpp
  wowfiles/Mddf.cpp
  wowfiles/Mh2o.cpp
  wowfiles/Mhdr.cpp  
  wowfiles/Mmid.cpp
  wowfiles/Mmdx.cpp  
  wowfiles/Modf.cpp
  wowfiles/Mphd.cpp
  wowfiles/Mwid.cpp  
  wowfiles/Mwmo.cpp    
  wowfiles/Wdl.cpp  
  wowfiles/Wdt.cpp
  
  main.cpp
  )

set ( gp_headers
  utilities/Utilities.h
  
  wowfiles/alpha/AdtAlpha.h
  wowfiles/alpha/WdtAlpha.h
  wowfiles/alpha/McnkAlpha.h
  wowfiles/alpha/McnrAlpha.h
  wowfiles/alpha/McvtAlpha.h
  wowfiles/alpha/Mdnm.h
  wowfiles/alpha/Monm.h
  wowfiles/alpha/MphdAlpha.h
  wowfiles/alpha/MainAlpha.h

  wowfiles/cataclysm/AdtCata.h
  wowfiles/cataclysm/AdtCataTerrain.h
  wowfiles/cataclysm/AdtCataTextures.h  
  wowfiles/cataclysm/AdtCataObjects.h    
  wowfiles/cataclysm/McnkCataTerrain.h
  wowfiles/cataclysm/McnkCataObjects.h
  wowfiles/cataclysm/McnkCataTextures.h
  wowfiles/cataclysm/McnrCata.h

  wowfiles/lichking/AdtLk.h
  wowfiles/lichking/McnkLk.h
  wowfiles/lichking/McnrLk.h

  wowfiles/Chunk.h
  wowfiles/ChunkHeaders.h  
  wowfiles/Main.h
  wowfiles/Mcal.h
  wowfiles/Mcin.h
  wowfiles/Mcrf.h
  wowfiles/Mcnk.h
  wowfiles/Mddf.h  
  wowfiles/Mh2o.h
  wowfiles/Mhdr.h
  wowfiles/Mmid.h
  wowfiles/Mmdx.h  
  wowfiles/Modf.h   
  wowfiles/Mphd.h
  wowfiles/Mwid.h 
  wowfiles/Mwmo.h   
  wowfiles/Wdl.h
  wowfiles/Wdt.h
  wowfiles/WowChunkedFormat.h
  )

include_directories (${CMAKE_CURRENT_SOURCE_DIR})

add_executable( gillijimproject
  ${gp_headers}
  ${gp_sources}  
  )
