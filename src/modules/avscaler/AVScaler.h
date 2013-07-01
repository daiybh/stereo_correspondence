/*!
 * @file 		AVScaler.h
 * @author 		Zdenek Travnicek
 * @date 		24.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef AVSCALER_H_
#define AVSCALER_H_
#include "yuri/libav/AVCodecBase.h"
extern "C" {
	#include <libswscale/swscale.h>
}

namespace yuri
{
namespace video
{

class AVScaler:public AVCodecBase
{
public:
	AVScaler(log::Log &_log, core::pwThreadBase parent, core::Parameters& parameters);
	virtual ~AVScaler();
	//static core::pBasicIOThread generate(log::Log &_log,core::pwThreadBase parent, core::Parameters& parameters);
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();
	static bool configure_converter(core::Parameters& parameters,
				long format_in, long format_out);

	bool set_output_format(int w, int h, long fmt);
	//virtual void connect_in(int index,core::pBasicPipe pipe);
	virtual bool synchronous_scale(shared_ptr<AVFrame> fr,int w, int h, PixelFormat fmt, int pts);
	void out_close() { if (out[0]) out[0]->close(); }
	//bool out_is_full() { if (out[0]) return out[0]->is_full(); return false; }
	static void av_ctx_deleter(SwsContext *ctx);
	static std::set<long> get_supported_formats();
protected:
	void run();

	bool do_recheck_conversions();
	void do_create_contexts();
	void scale_frame();
	void do_scale_frame();
	bool do_fetch_frame();
	bool do_check_input_frame();
	void do_delete_contexts();
	void do_output_frame(core::pBasicFrame frame);
	bool do_prescale_checks();
	virtual bool step();
protected:
	PixelFormat f_in, f_out;
	yuri::format_t format_in, format_out;
	yuri::ssize_t w_in, h_in, w_out, h_out;
	bool scaling, transforming, valid_contexts;
	yuri::mutex scaler_lock;
	shared_ptr<SwsContext> scale_ctx, transform_ctx;
	shared_ptr<AVPicture> pix_out, pix_inter;
	core::pBasicFrame frm_out, frm_inter;

	bool input_pipe_connected,scaling_disabled;
	yuri::size_t pts, duration;
	core::pBasicFrame frame;
	shared_ptr<AVFrame> synch_frame;
};

}
}

#endif /*AVSCALER_H_*/
