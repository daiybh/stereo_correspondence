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
#include "yuri/core/BasicIOFilter.h"
namespace yuri
{

namespace video
{

class AVEncoder: public core::BasicIOFilter, public AVCodecBase
{
public:
	virtual ~AVEncoder();
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();

	bool init_encoder();
	void run();
	virtual bool set_param(const core::Parameter &param);
protected:
	AVEncoder(log::Log &_log, core::pwThreadBase parent,core::Parameters &parameters) IO_THREAD_CONSTRUCTOR;
	core::pBasicFrame  encode_frame(const core::pBasicFrame& frame_in);
//	virtual bool step();
	core::pBasicFrame do_simple_single_step(const core::pBasicFrame& frame);
	shared_ptr<AVFrame> frame;
	std::vector<yuri::ubyte_t> buffer;
	yuri::size_t buffer_size;
	float time_step;
	yuri::size_t width, height;
	//CodecID codec;
	static std::set<yuri::format_t> get_supported_input_formats();
	static std::set<yuri::format_t> get_supported_output_formats();
	std::set<yuri::format_t> supported_formats_for_current_codec;
};

}

}

#endif /*ENCODER_H_*/
