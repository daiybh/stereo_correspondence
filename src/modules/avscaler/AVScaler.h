#ifndef AVSCALER_H_
#define AVSCALER_H_
#include "yuri/libav/AVCodecBase.h"
#include "yuri/config/Config.h"
#include "yuri/config/RegisteredClass.h"

extern "C" {
	#include <libswscale/swscale.h>
}
namespace yuri
{
namespace video
{
using namespace yuri::io;
using namespace yuri::config;

class AVScaler:public AVCodecBase
{
public:
	AVScaler(Log &_log, pThreadBase parent);
	virtual ~AVScaler();
	static shared_ptr<BasicIOThread> generate(Log &_log,pThreadBase parent,Parameters& parameters) throw (Exception);
	static shared_ptr<Parameters> configure();
	static bool configure_converter(Parameters& parameters,
				long format_in, long format_out) throw (Exception);

	bool set_output_format(int w, int h, long fmt);
	//virtual void connect_in(int index,shared_ptr<BasicPipe> pipe);
	virtual bool synchronous_scale(shared_ptr<AVFrame> fr,int w, int h, PixelFormat fmt, int pts);
	void out_close() { if (out[0]) out[0]->close(); }
	//bool out_is_full() { if (out[0]) return out[0]->is_full(); return false; }
	static void av_ctx_deleter(SwsContext *ctx);
	static set<long> get_supported_formats();
protected:
	void run();

	bool do_recheck_conversions();
	void do_create_contexts();
	void scale_frame();
	void do_scale_frame();
	bool do_fetch_frame();
	bool do_check_input_frame();
	void do_delete_contexts();
	void do_output_frame(shared_ptr<BasicFrame> frame);
	bool do_prescale_checks();
	virtual bool step();
protected:
	PixelFormat f_in, f_out;
	yuri::format_t format_in, format_out;
	yuri::ssize_t w_in, h_in, w_out, h_out;
	bool scaling, transforming, valid_contexts;
	boost::mutex scaler_lock;
	shared_ptr<SwsContext> scale_ctx, transform_ctx;
	shared_ptr<AVPicture> pix_out, pix_inter;
	shared_ptr<BasicFrame> frm_out, frm_inter;

	bool input_pipe_connected,scaling_disabled;
	yuri::size_t pts, duration;
	shared_ptr<BasicFrame> frame;
	shared_ptr<AVFrame> synch_frame;
};

}
}

#endif /*AVSCALER_H_*/
