#ifndef _WOWFILES_ALPHA_MCNKALPHA_H_
#define _WOWFILES_ALPHA_MCNKALPHA_H_

#include <vector>
#include <iostream>
#include <iostream>
#include <fstream>
#include <string>
#include <wowfiles/Chunk.h>
#include <wowfiles/ChunkHeaders.h>
#include <wowfiles/alpha/McvtAlpha.h>
#include <wowfiles/alpha/McnrAlpha.h>
#include <wowfiles/lichking/McnkLk.h>
#include <wowfiles/Mcal.h>
#include <wowfiles/Mcrf.h>

class McnkAlpha : public Mcnk
{
  public:

    McnkAlpha(std::ifstream & wdtAlphaFile, int offsetInFile, const int headerSize, int adtNum);
    McnkLk toMcnkLk( std::vector<int> alphaM2Indices, std::vector<int> alphaWmoIndices ) const;

    friend std::ostream & operator<<(std::ostream & os, const McnkAlpha & mcnkAlpha);

  private:

    int adtNumber;
    McnkAlphaHeader mcnkAlphaHeader;
    McvtAlpha mcvt;
    McnrAlpha mcnrAlpha;
    Chunk mcly;
    Mcrf mcrf;
    Chunk mcsh;
    Mcal mcal;
    Chunk mclq;
};

#endif
