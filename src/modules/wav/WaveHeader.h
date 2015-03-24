/*!
 * @file 		WaveHeader.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		20.03.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef WAVEHEADER_H_
#define WAVEHEADER_H_
#include "yuri/core/utils/platform.h"
#include <cstdint>
PACK_START

namespace yuri {
namespace wav {

struct wav_header_t
{
	char cID[4];
	uint32_t cSize;
	char wavID[4];
	char subID[4];
	uint32_t subSize;
	uint16_t fmt;
	uint16_t channels;
	uint32_t rate;
	uint32_t byte_rate;
	uint16_t block_align;
	uint16_t bps;
	char dataID[4];
	uint32_t dataSize;
	wav_header_t(uint16_t channels=2,uint32_t rate=44100,uint16_t bps=16,bool le=true):
			cSize(0),
			subSize(16),fmt(1),channels(channels),rate(rate),
			byte_rate((rate*channels*bps)>>3),block_align((channels*bps)>>3),bps(bps),
			dataSize(0)
	{
		std::copy_n("RIFF",4,cID);
		if (!le) cID[3]='X';
		std::copy_n("WAVE",4,wavID);
		std::copy_n("fmt ",4,subID);
		std::copy_n("data",4,dataID);
	}
	void add_size(uint32_t size) { dataSize+=size;cSize=36+dataSize; }

} PACK_END;

}
}
#endif /* WAVEHEADER_H_ */
