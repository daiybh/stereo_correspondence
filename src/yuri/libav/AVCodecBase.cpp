/*!
 * @file 		AVCodecBase.cpp
 * @author 		Zdenek Travnicek
 * @date 		24.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "AVCodecBase.h"

#include <boost/algorithm/string.hpp>
#include <boost/assign.hpp>
namespace yuri
{
namespace video
{

boost::mutex AVCodecBase::avcodec_lock;
bool AVCodecBase::avcodec_initialized = false;
map<yuri::format_t, CodecID> AVCodecBase::yuri_codec_map=boost::assign::map_list_of
		(YURI_VIDEO_MPEG2, 			CODEC_ID_MPEG2VIDEO)
		(YURI_VIDEO_MPEG1, 			CODEC_ID_MPEG1VIDEO)
		(YURI_VIDEO_HUFFYUV, 		CODEC_ID_HUFFYUV)
		(YURI_VIDEO_DV, 			CODEC_ID_DVVIDEO)
		(YURI_VIDEO_MPEGTS,			CODEC_ID_MPEG2TS)
		(YURI_VIDEO_MJPEG,			CODEC_ID_MJPEG)
		(YURI_VIDEO_H264,			CODEC_ID_H264)
		(YURI_VIDEO_FLASH,			CODEC_ID_FLASHSV2)
		(YURI_VIDEO_DIRAC,			CODEC_ID_DIRAC)
		(YURI_VIDEO_H263,			CODEC_ID_H263)
		(YURI_VIDEO_H263PLUS,		CODEC_ID_H263P)
		(YURI_VIDEO_THEORA,			CODEC_ID_THEORA)
		(YURI_VIDEO_VP8,			CODEC_ID_VP8);

map<yuri::format_t, PixelFormat> AVCodecBase::yuri_pixel_map=boost::assign::map_list_of
		(YURI_FMT_RGB,	 			PIX_FMT_RGB24)
		(YURI_FMT_BGR,	 			PIX_FMT_BGR24)
		(YURI_FMT_RGBA,	 			PIX_FMT_RGBA)
		(YURI_FMT_YUV422,			PIX_FMT_YUYV422)
		(YURI_FMT_YUV420_PLANAR,	PIX_FMT_YUV420P)
		(YURI_FMT_YUV422_PLANAR,	PIX_FMT_YUV422P)
		(YURI_FMT_YUV444_PLANAR,	PIX_FMT_YUV444P);
map<PixelFormat, PixelFormat> AVCodecBase::av_identity=boost::assign::map_list_of
		(PIX_FMT_YUVJ420P,			PIX_FMT_YUV420P)
		(PIX_FMT_YUVJ422P,			PIX_FMT_YUV422P);
AVCodecBase::AVCodecBase(Log &_log, pThreadBase parent, string id, yuri::sint_t inp, yuri::sint_t outp)
	IO_THREAD_CONSTRUCTOR:
	BasicIOThread(_log,parent,inp,outp,id),cc(0),c(0),codec_id(CODEC_ID_NONE),
	current_format(YURI_FMT_NONE),opened(false)
{
	initialize_avcodec();
}

AVCodecBase::~AVCodecBase()
{
}

/// Initializes CodecContext and Codec
///
/// If there's no CodecContext, it is created and initialized using provided parameters.
/// Otherwise existing CodecContext is used and parameters are only applied to object, but NOT for CodecContext!
/// @param codec_type	- Codec type (AVMEDIA_TYPE_VIDEO etc.)
/// @param codec_id		- codec id (defined in avcodec.h)
/// @param width		- width of video
/// @param height		- height
/// @param bps			- bitrate
/// @param fps
/// @param fps_base
/// @return true on successful initialization of codec, false otherwise
bool AVCodecBase::init_codec(AVMediaType codec_type, int width, int height,
		int bps, int fps, int fps_base)
{
	this->codec_type=codec_type;
	this->codec_id=codec_id;
	this->width=width;
	this->height=height;
	this->bps=bps;
	this->fps=fps;
	this->fps_base=fps_base;

	if (!c) return false;
	if (!cc) {  // Allocate codec context if none is provided
		cc=avcodec_alloc_context3(c);
		if (!cc) {
			log[error] << "Failed to allocate codec context" << std::endl;
			return false;
		}
		cc->codec_id=(CodecID)codec_id;
		cc->codec_type=codec_type;
		cc->bit_rate=bps;
		cc->width=width;
		cc->height=height;
		cc->time_base.den=fps;
		cc->time_base.num=fps_base;
		cc->gop_size = 12;/*
		if (codec_id==CODEC_ID_HUFFYUV) {
			cc->pix_fmt = PIX_FMT_YUV422P;
		} else if (codec_id == CODEC_ID_MJPEG) {
			cc->pix_fmt = PIX_FMT_YUVJ422P;
		} else if (codec_id == CODEC_ID_DVVIDEO) {
			cc->pix_fmt = PIX_FMT_YUV420P;
		} else cc->pix_fmt = PIX_FMT_YUV420P;*/
		if (!c->pix_fmts || c->pix_fmts[0]==-1) {
			// No input pixel format supported? Probably in decoder...
			cc->pix_fmt = PIX_FMT_NONE;
		} else {
			cc->pix_fmt = c->pix_fmts[0];
		}
		if (cc->codec_id == CODEC_ID_MPEG2VIDEO) {
			cc->max_b_frames = 2;
		}
		if (cc->codec_id == CODEC_ID_MPEG1VIDEO){
			cc->mb_decision=2;
		}

	}
	current_format=yuri_pixelformat_from_av(cc->pix_fmt);
	if (opened) avcodec_close(cc);
	if (avcodec_open2(cc,c,0)<0) {
		log[error] << "Opening codec failed! (" << c << ", " << cc << ")" << std::endl;
		return false;
	}
	opened=true;
	return true;
}
bool AVCodecBase::find_decoder()
{
	if ((c=avcodec_find_decoder(codec_id))) return true;
	return false;
}

bool AVCodecBase::find_encoder()
{
	if ((c=avcodec_find_encoder(codec_id))) return true;
	return false;
}

AVFrame * AVCodecBase::alloc_picture(PixelFormat fmt,int w, int h)
{
	AVFrame *picture;
	uint8_t *picture_buf;
	int size;
 	picture = avcodec_alloc_frame();
	if (!picture) return NULL;
	size = avpicture_get_size(fmt, w, h);
	picture_buf = (uint8_t*)av_malloc(size);
	if (!picture_buf) {
		av_free(picture);
		return 0;
	}
	avpicture_fill((AVPicture *)picture, picture_buf, fmt, w, h);
	return picture;


}


void AVCodecBase::free_picture(AVFrame *pix)
{
	if (!pix) return;
	/// \todo I should free even the buffer inside...
	av_free(pix);

}


/// Packs frame from planes into single memory area
///
/// source picture is defined by pixel format, width, height, array of pointers to data planes and linesizes
/// Output is stores in memory and in ls, there are linesizes copied again.
/// @param ftm
/// @param width
/// @param height
/// @param size
/// @param memory
/// @param ls
/// @param data
/// @param linesize
void AVCodecBase::do_pack_frame_data(PixelFormat fmt, int width, int height,
		yuri::size_t *size, char **memory, uint8_t **data, int *linesize)
{
	(*size) = avpicture_get_size(fmt,width,height);
	(*memory) = new char[*size];
	AVPicture pic;
	memcpy (pic.linesize,linesize,sizeof(int)*4);
	memcpy (pic.data,data,sizeof(uint8_t*)*4);
	avpicture_layout(&pic,fmt, width, height, (unsigned char*)*memory, *size);
}
void AVCodecBase::do_map_frame_data(PixelFormat fmt, int width, int height,
		uint8_t **planes, uint8_t *memory, int *linesizes)
{
	AVPicture pic;
	avpicture_fill(&pic, memory, fmt, width, height);
	memcpy(planes,pic.data,sizeof(uint8_t*)*4);
	memcpy(linesizes,pic.linesize,sizeof(int)*4);
}

shared_ptr<AVPicture> AVCodecBase::convert_to_avpicture(shared_ptr<BasicFrame> frame)
{
	assert(frame);
	shared_ptr<AVPicture> pic(new AVPicture);
	set_av_frame_or_picture(frame,pic);
	return pic;
}
shared_ptr<AVFrame> AVCodecBase::convert_to_avframe(shared_ptr<BasicFrame> frame)
{
	assert(frame);
	shared_ptr<AVFrame> frm (avcodec_alloc_frame(),AVCodecBase::av_frame_deleter);
	set_av_frame_or_picture(frame,frm);
	return frm;
}
CodecID AVCodecBase::get_codec_from_string(string codec)
{
	yuri::format_t fmt = BasicPipe::get_format_from_string(codec);
	if (!fmt) return CODEC_ID_NONE;
	return avcodec_from_yuri_format(fmt);
}


PixelFormat AVCodecBase::av_pixelformat_from_yuri(yuri::format_t format) throw (Exception)
{
	if (yuri_pixel_map.count(format)) return yuri_pixel_map[format];
	return PIX_FMT_NONE;
}
yuri::format_t AVCodecBase::yuri_pixelformat_from_av(PixelFormat format)throw (Exception)
{
	pair<yuri::format_t, PixelFormat> fp;
	PixelFormat pixfmt = format;
	if (av_identity.count(format)) pixfmt=av_identity[format];
	BOOST_FOREACH(fp, yuri_pixel_map) {
		if (fp.second==pixfmt) return fp.first;
	}
	return YURI_FMT_NONE;
}
yuri::format_t AVCodecBase::yuri_format_from_avcodec(CodecID codec) throw (Exception)
{
	pair<yuri::format_t, CodecID> fp;
	BOOST_FOREACH(fp, yuri_codec_map) {
		if (fp.second==codec) return fp.first;
	}
	return YURI_FMT_NONE;
}

CodecID AVCodecBase::avcodec_from_yuri_format(yuri::format_t codec) throw (Exception)
{
	if (yuri_codec_map.count(codec)) return yuri_codec_map[codec];
	return CODEC_ID_NONE;
}

void AVCodecBase::av_frame_deleter(AVFrame *frame)
{
	av_free(frame);
}
yuri::size_t AVCodecBase::calculate_time(yuri::size_t timestamp, AVRational &base)
{
	return 1e6 * base.num * timestamp / base.den;
}
void AVCodecBase::initialize_avcodec()
{
	boost::mutex::scoped_lock l(avcodec_lock);
	if (avcodec_initialized) return;
	avcodec_register_all();
	avcodec_initialized = true;
}
}
}
// End of file

