/*!
 * @file 		JPEGEncoder.h
 * @author 		Zdenek Travnicek
 * @date 		29.7.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2009 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef JPEGENCODER_H_
#define JPEGENCODER_H_


#include <cstdio>
#include <jpeglib.h>
#include "yuri/core/BasicIOThread.h"

namespace yuri {

namespace jpg {


class JPEGEncoder: public core::BasicIOThread {
public:
	JPEGEncoder(log::Log &_log, core::pwThreadBase parent, int level=75, long buffer_size=1048576);
	virtual ~JPEGEncoder();
	static core::pBasicIOThread generate(log::Log &_log,core::pwThreadBase parent,
			core::Parameters& parameters);
//	static bool configure_converter(Parameters& parameters,
//			long format_in, long format_out) throw (Exception);
	static core::pParameters configure();
protected:
	virtual bool step();
	void setDestManager(jpeg_compress_struct* cinfo);
	void initDestination(j_compress_ptr cinfo);
	int emptyBuffer(j_compress_ptr cinfo);
static	void sInitDestination(j_compress_ptr cinfo);
static	int sEmptyBuffer(j_compress_ptr cinfo);
static	void sTermDestination(j_compress_ptr cinfo);
	yuri::size_t dumpData();
protected:
	core::pBasicFrame frame;
std::stringstream temp_data;
	int level;
	std::vector<char> buffer;
	long buffer_size, width, height;
};

}

}
#endif /* JPEGENCODER_H_ */
