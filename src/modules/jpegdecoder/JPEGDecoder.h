/*!
 * @file 		JPEGDecoder.h
 * @author 		Zdenek Travnicek
 * @date 		3.8.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2009 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef JPEGDECODER_H_
#define JPEGDECODER_H_


#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/CompressedVideoFrame.h"
#include <jpeglib.h>


namespace yuri {

namespace jpg {

class JPEGDecoder:public core::SpecializedIOFilter<core::CompressedVideoFrame> {
public:
	JPEGDecoder(const log::Log &_log, core::pwThreadBase parent, const core::Parameters& parameters);
	virtual ~JPEGDecoder() noexcept;
	//static core::pIOThread generate(log::Log &_log,core::pwThreadBase parent,core::Parameters& parameters);
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();

//	static bool validate(core::pFrame f);
	//virtual bool step();
	virtual core::pFrame do_special_single_step(const core::pCompressedVideoFrame& frame) override;
//	void forceLineWidthMult(int mult) { if (mult>1)line_width_mult = mult; else mult=1; }
private:
	bool set_param(const core::Parameter& param);
	void setDestManager(jpeg_decompress_struct* cinfo);
	static void initSrc(jpeg_decompress_struct* cinfo);
	//void initSource(jpeg_decompress_struct* cinfo);

	static int fillInput(jpeg_decompress_struct* cinfo);
	static void skipData(jpeg_decompress_struct* cinfo, long numbytes);
	static int resyncData(jpeg_decompress_struct* cinfo, int desired);
	static void termSource(jpeg_decompress_struct* cinfo);
	static void errorExit(jpeg_common_struct* cinfo);
	void abort();

	core::pCompressedVideoFrame frame;
	int line_width_mult;
	bool aborted;
	format_t format_;
	bool raw_;
	bool fast_;

};

}

}
#endif /* JPEGDECODER_H_ */
