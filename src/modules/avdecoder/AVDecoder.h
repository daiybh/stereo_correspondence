#ifndef DECODER_H_
#define DECODER_H_
#include "yuri/libav/AVCodecBase.h"
#include <yuri/config/Config.h>
namespace yuri
{
namespace video
{
using namespace yuri::io;
using namespace yuri::log;
using namespace yuri::config;

class AVDecoder: public AVCodecBase
{
public:

	virtual ~AVDecoder();
	IO_THREAD_GENERATOR_DECLARATION
	static shared_ptr<Parameters> configure();

	bool init_decoder(CodecID codec_id, int width, int height);
	bool init_decoder(AVCodecContext *cc);
	//virtual void run();
	float get_fps();
//	void force_synchronous_scaler(int w, int h, PixelFormat fmt);
	//virtual void connect_out(int index,Pipe *pipe);
	//virtual boost::thread* spawn_thread();
	bool regenerate_contexts(long format,yuri::size_t width, size_t height);
	virtual bool set_param(Parameter &param);
protected:
	AVDecoder(Log &_log, pThreadBase parent,Parameters &parameters) IO_THREAD_CONSTRUCTOR;
	bool decode_frame();
	void do_output_frame();
	virtual bool step();
protected:
	shared_ptr<AVFrame> frame;
	float time_step;
	yuri::size_t last_pts, first_pts;
//	shared_ptr<AVScaler> scaler;
	long decoding_format;
	shared_ptr<BasicFrame> input_frame;
	shared_ptr<BasicFrame> output_frame;
	bool use_timestamps;
	yuri::size_t first_out_pts;
	boost::posix_time::ptime first_time;
};

}
}
#endif /*DECODER_H_*/
