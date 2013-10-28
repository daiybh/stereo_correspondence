/*!
 * @file 		AVCodecBase.h
 * @author 		Zdenek Travnicek
 * @date 		24.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef AVCODECBASE_H_
#define AVCODECBASE_H_

#include "yuri/core/IOThread.h"
#include "yuri/core/BasicFrame.h"
#include "yuri/core/BasicPipe.h"
#include "yuri/core/pipe_types.h"
extern "C" {
	#include "libavcodec/avcodec.h"
	//include "avutil.h"
}
namespace yuri
{
namespace video
{
class AVCodecBase//: public core::IOThread
{
public:
	AVCodecBase(log::Log& log_);
	virtual ~AVCodecBase();
protected:
	bool init_codec(AVMediaType codec_type, int width, int height, int bps, int fps, int fps_base);
	AVFrame * alloc_picture(PixelFormat fmt,int w, int h);
	void free_picture(AVFrame *pix);
	bool find_decoder();
	bool find_encoder();	
	void do_pack_frame_data(PixelFormat fmt, int width, int height, yuri::size_t *size, char **memory, uint8_t **data, int *linesize) DEPRECATED;
	void do_map_frame_data(PixelFormat fmt, int width, int height, uint8_t **planes, uint8_t *memory, int *linesizes) DEPRECATED;
	static void initialize_avcodec();
	static shared_ptr<AVFrame> convert_to_avframe(core::pBasicFrame frame);
	static shared_ptr<AVPicture> convert_to_avpicture(core::pBasicFrame  frame);
	template<typename T> static void set_av_frame_or_picture(core::pBasicFrame  frame, shared_ptr<T> av);
	//static shared_ptr<AVPicture> allocate_avpicture(long format);
	static CodecID get_codec_from_string(std::string codec);
	static PixelFormat av_pixelformat_from_yuri(yuri::format_t format);
	static yuri::format_t yuri_pixelformat_from_av(PixelFormat format);
	static yuri::format_t yuri_format_from_avcodec(CodecID codec);
	static CodecID avcodec_from_yuri_format(yuri::format_t codec);
	static void av_frame_deleter(AVFrame*);
	template<typename T> static void av_deleter(T *ptr) { if (ptr) av_free(ptr); }
	static yuri::size_t calculate_time(yuri::size_t timestamp, AVRational &base);
protected: 
	static yuri::mutex avcodec_lock;
	static bool avcodec_initialized;
	static std::map<yuri::format_t, CodecID> yuri_codec_map;
	static std::map<yuri::format_t, PixelFormat> yuri_pixel_map;
	// Used to convert between 'identical' format. TODO: needs better solution
	static std::map<PixelFormat, PixelFormat> av_identity;

	log::Log 				&log_;
	AVCodecContext			*cc;
	AVCodec 				*c;
	CodecID 				codec_id;
	yuri::format_t 			current_format;
	int 					width,
							height,
							bps,
							fps,
							fps_base;
	AVMediaType 			codec_type;
	bool 					opened;


};

template<typename T> void AVCodecBase::set_av_frame_or_picture(core::pBasicFrame  frame,shared_ptr<T> av)
{
	FormatInfo_t fmt = core::BasicPipe::get_format_info(frame->get_format());
	//log[info] << "Format " << fmt->name << " with " << fmt->planes << " planes" <<endl;
	assert(fmt->planes && fmt->planes == frame->get_planes_count());
	//yuri::size_t width = frame->get_width();
	yuri::size_t height = frame->get_height(), no_planes, bpplane;
	if (frame->get_planes_count() !=  fmt->planes) {
		// This should never happen. Something has gone wrong.
//		throw("bad plane numbers (expected: " <<fmt->planes << ", got: " << frame->get_planes_count() << ")"  << endl;)
	}
	no_planes = std::min(frame->get_planes_count(), fmt->planes);
	bpplane = fmt->bpp;
	for (yuri::size_t i = 0; i < 4; ++i) {
		if (no_planes>1) bpplane= fmt->component_depths[i];
		if (i >= no_planes) {
			av->data[i]=0;
			av->linesize[i]=0;
		} else {
//			av->data[i]=(uint8_t*)((*frame)[i].data.get());
			av->data[i]=reinterpret_cast<uint8_t*>(PLANE_RAW_DATA(frame,i));

			//av->linesize[i]=fmt->bpp*width/fmt->plane_x_subs[i]/8;
			if (height) {
				//av->linesize[i]=(*frame)[i].get_size()/(height*fmt->plane_y_subs[i]);

				av->linesize[i]=(bpplane*frame->get_width()/fmt->plane_x_subs[i])>>3;
			} else {
				av->linesize[i]=0;
			}
			//cout << av->linesize[i] << endl;
		}
		//log[debug] << "set linesize[" << i<<"] to " << av->linesize[i] << endl;
	}
}


}
}
#endif /*AVCODECBASE_H_*/
