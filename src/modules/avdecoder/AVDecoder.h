/*!
 * @file 		AVDecoder.cpp
 * @author 		Zdenek Travnicek
 * @date 		24.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2010 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef DECODER_H_
#define DECODER_H_
#include "yuri/libav/AVCodecBase.h"
#include "yuri/core/BasicIOFilter.h"
namespace yuri
{
namespace video
{

class AVDecoder: public core::BasicIOFilter, public AVCodecBase
{
public:

	virtual ~AVDecoder();
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();

	bool init_decoder(CodecID codec_id, int width, int height);
	bool init_decoder(AVCodecContext *cc);
	//virtual void run();
	float get_fps();
//	void force_synchronous_scaler(int w, int h, PixelFormat fmt);
	//virtual void connect_out(int index,Pipe *pipe);
	//virtual boost::thread* spawn_thread();
	bool regenerate_contexts(long format,yuri::size_t width, size_t height);
	virtual bool set_param(const core::Parameter &param);
protected:
	AVDecoder(log::Log &_log, core::pwThreadBase parent, core::Parameters &parameters) IO_THREAD_CONSTRUCTOR;
	core::pBasicFrame decode_frame(const core::pBasicFrame& frame);
	core::pBasicFrame do_output_frame();
	core::pBasicFrame do_simple_single_step(const core::pBasicFrame& frame);
//	virtual bool step();
protected:
	shared_ptr<AVFrame> frame;
	float time_step;
	yuri::size_t last_pts, first_pts;
//	shared_ptr<AVScaler> scaler;
	long decoding_format;
	core::pBasicFrame input_frame;
	core::pBasicFrame output_frame;
	bool use_timestamps;
	yuri::size_t first_out_pts;
	time_value first_time;
};

}
}
#endif /*DECODER_H_*/
