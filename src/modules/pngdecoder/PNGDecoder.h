/*!
 * @file 		PNGDecoder.h
 * @author 		Zdenek Travnicek
 * @date 		27.7.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2009 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef PNGDECODE_H_
#define PNGDECODE_H_
#include <yuri/core/BasicIOThread.h>
#include <png.h>
namespace yuri {

namespace png {

class PNGDecoder: public core::BasicIOThread {
public:
	PNGDecoder(log::Log &_log, core::pwThreadBase parent,core::Parameters& parameters);
	virtual ~PNGDecoder();
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();
	virtual bool step();
	static bool validatePng(core::pBasicFrame f);
protected:
	static void readData(png_structp pngPtr, png_bytep data, png_size_t length);
	void readData(png_bytep data, png_size_t length);
	long position;
	core::pBasicFrame f;
	//png_structp pngPtr;
	//png_infop infoPtr;
};

}

}

#endif /* PNGDECODE_H_ */
