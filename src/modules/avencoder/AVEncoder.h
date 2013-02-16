/*!
 * @file 		AVEncoder.h
 * @author 		Zdenek Travnicek
 * @date 		24.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef ENCODER_H_
#define ENCODER_H_
#include "yuri/libav/AVCodecBase.h"
#include "yuri/config/Config.h"
#include "yuri/config/RegisteredClass.h"

namespace yuri
{

namespace video
{
using namespace yuri::config;
class AVEncoder: public AVCodecBase
{
public:
	virtual ~AVEncoder();
	IO_THREAD_GENERATOR_DECLARATION
	static shared_ptr<Parameters> configure();

	bool init_encoder();
	void run();
	virtual bool set_param(Parameter &param);
protected:
	AVEncoder(Log &_log, pThreadBase parent,Parameters &parameters) IO_THREAD_CONSTRUCTOR;
	void encode_frame();
	virtual bool step();
	shared_ptr<AVFrame> frame;
	shared_array<yuri::ubyte_t> buffer;
	yuri::size_t buffer_size;
	float time_step;
	yuri::size_t width, height;
	//CodecID codec;
	static set<yuri::format_t> get_supported_input_formats();
	static set<yuri::format_t> get_supported_output_formats();
	set<yuri::format_t> supported_formats_for_current_codec;
};

}

}

#endif /*ENCODER_H_*/
