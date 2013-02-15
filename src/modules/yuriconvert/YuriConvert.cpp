/*
 * YuriConvertor.cpp
 *
 *  Created on: Aug 13, 2010
 *      Author: neneko
 */
#ifdef _WIN32
#define WIN32_MEAN_AND_LEAN
#define NOMINMAX
#endif
#include "YuriConvert.h"
#include <boost/assign.hpp>
#ifdef YURI_HAVE_LIBMVTP
#include "libMVTP/MVTPConvert.h"
#endif
#ifdef YURI_HAVE_CUDA
	bool YuriConvertRGB24_YUV20(const char *src, char *dest, void *cuda_src, void *cuda_dest,unsigned int num, float Wb, float Wr);
	bool YuriConvertYUV16_RGB24(const char *src, char *dest, void *cuda_src, void *cuda_dest,unsigned int num, float Wb, float Wr);
	void *CudaAlloc(unsigned int size);
	void CudaDealloc(void *mem);
#endif
namespace yuri {

namespace video {

REGISTER("yuri_convert",YuriConvertor)

bool YuriConvertor::tables_initialized;
yuri::ubyte_t YuriConvertor::sixteenbits_yc_b0[256];
yuri::ubyte_t YuriConvertor::sixteenbits_yc_b1[65536];
yuri::ubyte_t YuriConvertor::sixteenbits_yc_b2[256];
yuri::ubyte_t YuriConvertor::sixteenbits_yc_b0b[256];
yuri::ubyte_t YuriConvertor::sixteenbits_yc_b1b[65536];
yuri::ubyte_t YuriConvertor::sixteenbits_yc_b2b[256];

// 128MB of ancillary tables for RGB24->YCbCr
yuri::ubyte_t YuriConvertor::rgb_y0[1<<24];
yuri::ubyte_t YuriConvertor::rgb_y1[1<<24];
yuri::ubyte_t YuriConvertor::rgb_y2[1<<24];
yuri::ubyte_t YuriConvertor::rgb_y3[1<<24];
yuri::ushort_t YuriConvertor::rgb_cr[1<<24];
yuri::ushort_t YuriConvertor::rgb_cb[1<<24];

boost::mutex YuriConvertor::tables_lock;


IO_THREAD_GENERATOR(YuriConvertor)

using boost::iequals;
template<> shared_ptr<BasicFrame> YuriConvertor::convert<YURI_FMT_YUV422,YURI_FMT_V210_MVTP,false>(shared_ptr<BasicFrame> frame);
template<> shared_ptr<BasicFrame> YuriConvertor::convert<YURI_FMT_RGB,YURI_FMT_V210_MVTP,false>(shared_ptr<BasicFrame> frame);
template<> shared_ptr<BasicFrame> YuriConvertor::convert<YURI_FMT_V210_MVTP,YURI_FMT_YUV422,false>(shared_ptr<BasicFrame> frame);
#ifdef YURI_HAVE_LIBMVTP
template<> shared_ptr<BasicFrame> YuriConvertor::convert<YURI_FMT_V210,YURI_FMT_V210_MVTP,false>(shared_ptr<BasicFrame> frame);
template<> shared_ptr<BasicFrame> YuriConvertor::convert<YURI_FMT_V210_MVTP,YURI_FMT_V210,false>(shared_ptr<BasicFrame> frame);
#endif
#ifdef YURI_HAVE_CUDA
template<> shared_ptr<BasicFrame> YuriConvertor::convert<YURI_FMT_YUV422,YURI_FMT_RGB,true>(shared_ptr<BasicFrame> frame);
template<> shared_ptr<BasicFrame> YuriConvertor::convert<YURI_FMT_RGB,YURI_FMT_V210_MVTP,true>(shared_ptr<BasicFrame> frame);
#endif

std::map<std::pair<yuri::format_t, yuri::format_t>, YuriConvertor::converter_t>
YuriConvertor::converters
#ifndef _WIN32
=boost::assign::map_list_of<std::pair<yuri::format_t, yuri::format_t>, YuriConvertor::converter_t>
		(std::make_pair(YURI_FMT_YUV422,	YURI_FMT_V210_MVTP),&YuriConvertor::convert<YURI_FMT_YUV422, 	YURI_FMT_V210_MVTP,	false>)
		(std::make_pair(YURI_FMT_RGB,		YURI_FMT_V210_MVTP),&YuriConvertor::convert<YURI_FMT_RGB, 		YURI_FMT_V210_MVTP,	false>)
		(std::make_pair(YURI_FMT_V210_MVTP, YURI_FMT_YUV422),	&YuriConvertor::convert<YURI_FMT_V210_MVTP,	YURI_FMT_YUV422,	false>)
#endif
#ifdef YURI_HAVE_LIBMVTP
		(std::make_pair(YURI_FMT_V210, 		YURI_FMT_V210_MVTP),&YuriConvertor::convert<YURI_FMT_V210,		YURI_FMT_V210_MVTP,	false>)
		(std::make_pair(YURI_FMT_V210_MVTP, 	YURI_FMT_V210),	    &YuriConvertor::convert<YURI_FMT_V210_MVTP,	YURI_FMT_V210,	false>)
#endif
;
#ifdef YURI_HAVE_CUDA
std::map<std::pair<yuri::format_t, yuri::format_t>, YuriConvertor::converter_t>
YuriConvertor::cuda_converters=boost::assign::map_list_of<std::pair<yuri::format_t, yuri::format_t>, YuriConvertor::converter_t>
		(std::make_pair(YURI_FMT_YUV422,	YURI_FMT_RGB),		&YuriConvertor::convert<YURI_FMT_YUV422, 	YURI_FMT_RGB,		true>)
		(std::make_pair(YURI_FMT_RGB,		YURI_FMT_V210_MVTP),&YuriConvertor::convert<YURI_FMT_RGB, 		YURI_FMT_V210_MVTP,	true>)
;
#endif
shared_ptr<Parameters> YuriConvertor::configure()
{
	shared_ptr<Parameters> p = BasicIOThread::configure();
	//(*p)["alternative_tables"]["Use alternative pregenerated tables (not tested)"]=false;
	(*p)["colorimetry"]["Colorimetry to use when converting from RGB (BT709, BT601)"]="BT709";
#ifdef YURI_HAVE_CUDA
	(*p)["cuda"]["Use cuda"]=true;
#endif
	(*p)["format"]["Output format (YCrCb20, YUV422)"]=std::string("YCrCb20");
	p->add_input_format(YURI_FMT_YUV422);
	p->add_input_format(YURI_FMT_RGB24);
	p->add_output_format(YURI_FMT_V210_MVTP);
	p->add_output_format(YURI_FMT_RGB24);
	p->add_converter(YURI_FMT_YUV422, YURI_FMT_V210_MVTP, 0, false);
	return p;
}

YuriConvertor::YuriConvertor(Log &log_, pThreadBase parent,Parameters& parameters) IO_THREAD_CONSTRUCTOR
	:BasicIOThread(log_,parent,1,1,"YuriConv"),allocated_size(0)
{
	IO_THREAD_INIT("Yuri Convert")
	//alternative_tables = params["alternative_tables"].get<bool>();
	switch (colorimetry) {
		case YURI_COLORIMETRY_REC601:
			Wr = 0.299f;
			Wb = 0.114f;
			break;
		case YURI_COLORIMETRY_REC709:
		default:		// If anything goes wrong, let's fall back to REC709
			Wr = 0.2126f;
			Wb = 0.0722f;
			break;
	}

	generate_tables();
}

YuriConvertor::~YuriConvertor() {
}

bool YuriConvertor::step()
{
	if (!in[0] || in[0]->is_empty()) return true;
	shared_ptr<BasicFrame> frame = in[0]->pop_frame();
	if (!frame) return true;
	shared_ptr<BasicFrame> outframe;
	std::pair<yuri::format_t, yuri::format_t> conv_pair = std::make_pair(frame->get_format(), format);
	YuriConvertor::converter_t converter = 0;
#ifdef YURI_HAVE_CUDA
	if (cuda_converters.count(conv_pair)) converter = cuda_converters[conv_pair];
	else
#endif
	if (converters.count(conv_pair)) converter = converters[conv_pair];
	if (converter) outframe= (this->*converter)(frame);
	else return true;
	if (outframe) {
		outframe->set_info(frame->get_info());
		push_video_frame (0,outframe,format, frame->get_width(), frame->get_height(),frame->get_pts(),frame->get_duration(),frame->get_dts());
	}
	return true;
}

template<> shared_ptr<BasicFrame> YuriConvertor::convert<YURI_FMT_YUV422,YURI_FMT_V210_MVTP,false>(shared_ptr<BasicFrame> frame)
{
	shared_ptr<BasicFrame> output;
	if (frame->get_planes_count() != 1) {
		log[warning] << "Unsupported number of planes (" <<
				frame->get_planes_count()<< ") in input frame!" << std::endl;
		return output;
	}
	yuri::size_t in_size = (*frame)[0].size;
	yuri::size_t width = frame->get_width();
	yuri::size_t height = frame->get_height();
	if (width * height * 2 != in_size) {
		log [warning] << "input frame has wrong size. Expected " <<
				width * height * 2 << ", got " << in_size << std::endl;
		return output;
	}
	if (width % 2) {
		log[warning] << "Wrong line width (" << width << ")" << std::endl;
		return output;
	}
	output.reset(new BasicFrame(1));
	yuri::size_t out_size = width * height * 5 / 2;
	log[debug] << "Allocating " << out_size << " bytes for output frame" <<std::endl;
	shared_array<yuri::ubyte_t> data = allocate_memory_block(out_size);
	shared_array<yuri::ubyte_t> in_data = (*frame)[0].data;
	(*output)[0].set(data,out_size);
	for (yuri::size_t line = 0; line < height; ++ line) {
		yuri::ubyte_t *in_ptr =  &in_data[line*width*2];
		yuri::ubyte_t *out_ptr = &data[line*width*5>>1];
		unsigned int y0,y1;
		for (yuri::size_t pos=0; pos<width/2; ++ pos) {
			*out_ptr++=sixteenbits_yc_b0[*in_ptr&0xFF];
			y0=*in_ptr++;
			y0|=*in_ptr << 8;
			*out_ptr++=sixteenbits_yc_b1[y0&0xFFFF];
			*out_ptr=sixteenbits_yc_b2[*in_ptr++&0xFF];
			*out_ptr++|=sixteenbits_yc_b0b[*in_ptr&0xFF];
			y1=*in_ptr++;
			y1|=*in_ptr << 8;
			*out_ptr++=sixteenbits_yc_b1b[y1&0xFFFF];
			*out_ptr++=sixteenbits_yc_b2b[*in_ptr++&0xFF];
		}
	}
	return output;
}

#ifdef YURI_HAVE_LIBMVTP
#define V210_B0(x) (((x)&0x000003FF)>> 0)
#define V210_B1(x) (((x)&0x000FFC00)>>10)
#define V210_B2(x) (((x)&0x3FF00000)>>20)

template<> shared_ptr<BasicFrame> YuriConvertor::convert<YURI_FMT_V210,YURI_FMT_V210_MVTP,false>(shared_ptr<BasicFrame> frame)
{
	shared_ptr<BasicFrame> output;
	if (frame->get_planes_count() != 1) {
		log[warning] << "Unsupported number of planes (" <<
				frame->get_planes_count()<< ") in input frame!" << std::endl;
		return output;
	}
	yuri::size_t in_size = (*frame)[0].size;
	yuri::size_t width = frame->get_width();
	yuri::size_t height = frame->get_height();
	yuri::size_t line_width6 = width/6 + (width%6?1:0);
	// Special case (needed for blackink input ;)
	if (width==1280) line_width6 = 216;

	if (line_width6 * height * 16 > in_size) {
		log [warning] << "input frame has wrong size. Expected at least " <<
				(line_width6 * height * 16) << " ("<<line_width6<<"), got " << in_size << 
				" for frame " << width << "x" << height << std::endl;
		return output;
	}
//	if (width % 6) {
//		log[warning] << "Line width should be multiply of 6 (" << width << ")" << std::endl;
//		return output;
//	}
	output.reset(new BasicFrame(1));
	yuri::size_t out_size = width * height * 5 / 2;
	log[debug] << "Allocating " << out_size << " bytes for output frame" <<std::endl;
	shared_array<yuri::ubyte_t> data = allocate_memory_block(out_size+100); // Allocating a bit more just to make sure
	shared_array<yuri::ubyte_t> in_data = (*frame)[0].data;
	(*output)[0].set(data,out_size);
	// Line width /6 - number of samples per line
//	else if (width%6) line_width6 = 216;
	for (yuri::size_t line = 0; line < height; ++ line) {
//		yuri::size_t lpos = line*16*(width/6+width%6?1:0);
		yuri::size_t lpos = line*16*line_width6;
		yuri::uint_t *in_ptr =  reinterpret_cast<yuri::uint_t *>(&in_data[lpos]);
		MVTP::MVTPDataFormat *out_ptr = reinterpret_cast<MVTP::MVTPDataFormat *>(&data[line*width*5>>1]);
		for (yuri::size_t pos=0; pos<line_width6; ++pos) {
			out_ptr++->set_from_components(V210_B1(in_ptr[0]),V210_B0(in_ptr[1]),V210_B0(in_ptr[0]),V210_B2(in_ptr[0]));
			out_ptr++->set_from_components(V210_B2(in_ptr[1]),V210_B1(in_ptr[2]),V210_B1(in_ptr[1]),V210_B0(in_ptr[2]));
			out_ptr++->set_from_components(V210_B0(in_ptr[3]),V210_B2(in_ptr[3]),V210_B2(in_ptr[2]),V210_B1(in_ptr[3]));
			in_ptr+=4;
		}
	}
/*	yuri::uint_t *in_ptr =  reinterpret_cast<yuri::uint_t *>(&in_data[lpos]);
	MVTP::MVTPDataFormat *out_ptr = reinterpret_cast<MVTP::MVTPDataFormat *>(&data[line*width*5>>1]);

	for (yuri::size_t pos= 0; pos < width * height/2; pos+=6) {
//		yuri::size_t lpos = line*16*(width/6+width%6?1:0);
		for (yuri::size_t pos=0; pos<width; pos+=6) {
			out_ptr++->set_from_components(V210_B1(in_ptr[0]),V210_B0(in_ptr[1]),V210_B0(in_ptr[0]),V210_B2(in_ptr[0]));
			out_ptr++->set_from_components(V210_B2(in_ptr[1]),V210_B1(in_ptr[2]),V210_B1(in_ptr[1]),V210_B0(in_ptr[2]));
			out_ptr++->set_from_components(V210_B0(in_ptr[3]),V210_B2(in_ptr[3]),V210_B2(in_ptr[2]),V210_B1(in_ptr[3]));
			in_ptr+=4;
		}
	}*/
	return output;
}

#define V210_ENC(a,b,c) ((a&0x3FF)<<20)|((b&0x3FF)<<10)|((c&0x3FF)<<0)

template<> shared_ptr<BasicFrame> YuriConvertor::convert<YURI_FMT_V210_MVTP,YURI_FMT_V210,false>(shared_ptr<BasicFrame> frame)
{
	shared_ptr<BasicFrame> output;
	if (frame->get_planes_count() != 1) {
		log[warning] << "Unsupported number of planes (" <<
				frame->get_planes_count()<< ") in input frame!" << std::endl;
		return output;
	}
	yuri::size_t in_size = (*frame)[0].size;
	yuri::size_t width = frame->get_width();
	yuri::size_t height = frame->get_height();
	yuri::size_t line_width6 = width/6 + (width%6?1:0);
	// Special case (needed for blackink input ;)
	if (width==1280) line_width6 = 216;

/*	if (line_width6 * height * 16 > in_size) {
		log [warning] << "input frame has wrong size. Expected at least " <<
				(line_width6 * height * 16) << " ("<<line_width6<<"), got " << in_size << 
				" for frame " << width << "x" << height << std::endl;
		return output;
	}*/
	output.reset(new BasicFrame(1));
	yuri::size_t out_size = line_width6 * height * 16;
	log[debug] << "Allocating " << out_size << " bytes for output frame" <<std::endl;
	shared_array<yuri::ubyte_t> data = allocate_memory_block(out_size+100); // Allocating a bit more just to make sure
	shared_array<yuri::ubyte_t> in_data = (*frame)[0].data;
	(*output)[0].set(data,out_size);
	uint16_t yy[6];
	uint16_t cc[6];
	for (yuri::size_t line = 0; line < height; ++ line) {

		MVTP::MVTPDataFormat *in_ptr = reinterpret_cast<MVTP::MVTPDataFormat *>(&in_data[line*width*5>>1]);
		yuri::size_t lpos = line*16*line_width6;
		yuri::uint_t *out_ptr =  reinterpret_cast<yuri::uint_t *>(&data[lpos]);
		
		
		for (yuri::size_t pos=0; pos<line_width6; ++pos) {
			in_ptr++->split_components(&yy[0],&cc[0]);
			in_ptr++->split_components(&yy[2],&cc[2]);
			in_ptr++->split_components(&yy[4],&cc[4]);

			*out_ptr++=V210_ENC(cc[0],yy[0],cc[1]);
			*out_ptr++=V210_ENC(yy[1],cc[2],yy[2]);
			*out_ptr++=V210_ENC(cc[3],yy[3],cc[4]);
			*out_ptr++=V210_ENC(yy[4],cc[5],yy[5]);
//			out_ptr+=4;
		}
	}
	return output;
}


#endif
template<> shared_ptr<BasicFrame> YuriConvertor::convert<YURI_FMT_RGB,YURI_FMT_V210_MVTP,false>(shared_ptr<BasicFrame> frame)
{
	shared_ptr<BasicFrame> output;
	// Verify input frame
	if (frame->get_planes_count() != 1) {
		log[warning] << "Unsupported number of planes (" <<
				frame->get_planes_count()<< ") in input frame!" << std::endl;
		return output;
	}
	yuri::size_t in_size = (*frame)[0].size;
	yuri::size_t width = frame->get_width();
	yuri::size_t height = frame->get_height();
	if (width * height * 3 != in_size) {
		log [warning] << "input frame has wrong size. Expected " <<
				width * height * 2 << ", got " << in_size << std::endl;
		return output;
	}
	if (width % 2) {
		log[warning] << "Wrong line width (" << width << ")" << std::endl;
		return output;
	}
	output.reset(new BasicFrame(1));
	yuri::size_t out_size = width * height * 5 / 2;
	log[debug] << "Allocating " << out_size << " bytes for output frame" <<std::endl;
	shared_array<yuri::ubyte_t> data = allocate_memory_block(out_size);
	shared_array<yuri::ubyte_t> in_data = (*frame)[0].data;
	(*output)[0].set(data,out_size);
	for (yuri::size_t line = 0; line < height; ++ line) {
		yuri::ubyte_t *in_ptr =  &in_data[line*width*3];
		yuri::ubyte_t *out_ptr = &data[line*width*5>>1];
		yuri::uint_t rgb1, rgb2, cb, cr;
		for (yuri::size_t pos=0; pos<width/2; ++ pos) {
			rgb1=*in_ptr++;
			for (yuri::ubyte_t i=0;i<2;++i) rgb1=(rgb1<<8)|(*in_ptr++&0xFF);
			rgb2=*in_ptr++;
			for (yuri::ubyte_t i=0;i<2;++i) rgb2=(rgb2<<8)|(*in_ptr++&0xFF);
			//rgb1=in_ptr[0]<<16|in_ptr[1]<<8|in_ptr[2];
			//rgb2=in_ptr[3]<<16|in_ptr[4]<<8|in_ptr[5];
			//in_ptr+=6;
//			rgb1=*(reinterpret_cast<yuri::uint_t*>(in_ptr))>>8;
//			rgb2=*(reinterpret_cast<yuri::uint_t*>(in_ptr+3))>>8;
//			in_ptr+=6;
			cr=(rgb_cr[rgb1]+rgb_cr[rgb2])>>1;
			cb=(rgb_cb[rgb1]+rgb_cb[rgb2])>>1;

			*out_ptr++=rgb_y0[rgb1];
			*out_ptr++=rgb_y1[rgb1] | ((cr << 2)&0xfc);
			*out_ptr++=((cr >> 6) &0x0F) | rgb_y2[rgb2];
			*out_ptr++=rgb_y3[rgb2] | ((cb <<6) & 0xC0);
			*out_ptr++=(cb>>2) & 0xff;
		}
	}
	return output;
}


template<> shared_ptr<BasicFrame> YuriConvertor::convert<YURI_FMT_RGB,YURI_FMT_V210_MVTP,true>(shared_ptr<BasicFrame> frame)
{
#ifdef YURI_HAVE_CUDA
	shared_ptr<BasicFrame> output;
	// Verify input frame
	if (frame->get_planes_count() != 1) {
		log[warning] << "Unsupported number of planes (" <<
				frame->get_planes_count()<< ") in input frame!" << std::endl;
		return output;
	}
	yuri::size_t in_size = (*frame)[0].size;
	yuri::size_t width = frame->get_width();
	yuri::size_t height = frame->get_height();
	if (width * height * 3 != in_size) {
		log [warning] << "input frame has wrong size. Expected " <<
				width * height * 2 << ", got " << in_size << std::endl;
		return output;
	}
	if (width % 2) {
		log[warning] << "Wrong line width (" << width << ")" << std::endl;
		return output;
	}
	output.reset(new BasicFrame(1));
	yuri::size_t out_size = width * height * 5 / 2;
	if (allocated_size != in_size) {
		cuda_src = cuda_alloc(in_size);
		cuda_dest = cuda_alloc(out_size);
	}
	log[debug] << "Allocating " << out_size << " bytes for output frame" <<std::endl;
	shared_array<yuri::ubyte_t> data = allocate_memory_block(out_size);
	shared_array<yuri::ubyte_t> in_data = (*frame)[0].data;
	(*output)[0].set(data,out_size);
	YuriConvertRGB24_YUV20(reinterpret_cast<const char*>(in_data.get()),reinterpret_cast<char*>(data.get()),cuda_src.get(),cuda_dest.get(),width*height,Wb,Wr);
	return output;
#else
	log[warning] << "CUDA optimizations not compiled in. Falling back to CPU version" << std::endl;
	return convert<YURI_FMT_RGB,YURI_FMT_V210_MVTP,false>(frame);
#endif
}

template<> shared_ptr<BasicFrame> YuriConvertor::convert<YURI_FMT_YUV422,YURI_FMT_RGB,true>(shared_ptr<BasicFrame> frame)
{
	shared_ptr<BasicFrame> output;
#ifdef YURI_HAVE_CUDA
	// Verify input frame
	if (frame->get_planes_count() != 1) {
		log[warning] << "Unsupported number of planes (" <<
				frame->get_planes_count()<< ") in input frame!" << std::endl;
		return output;
	}
	yuri::size_t in_size = PLANE_SIZE(frame,0);
	yuri::size_t width = frame->get_width();
	yuri::size_t height = frame->get_height();
	if (width * height * 2 != in_size) {
		log [warning] << "input frame has wrong size. Expected " <<
				width * height * 2 << ", got " << in_size << std::endl;
		return output;
	}
	if (width % 2) {
		log[warning] << "Wrong line width (" << width << ")" << std::endl;
		return output;
	}
	output = allocate_empty_frame(YURI_FMT_RGB, width, height);
	yuri::size_t out_size = width*height*3;
	if (allocated_size != in_size) {
		cuda_src = cuda_alloc(in_size);
		cuda_dest = cuda_alloc(out_size);
	}

	//log[debug] << "Allocating " << out_size << " bytes for output frame" <<std::endl;

	shared_array<yuri::ubyte_t> data = PLANE_DATA(output,0);
	shared_array<yuri::ubyte_t> in_data = PLANE_DATA(frame, 0);

	YuriConvertYUV16_RGB24(reinterpret_cast<const char*>(in_data.get()),reinterpret_cast<char*>(data.get()),cuda_src.get(),cuda_dest.get(),width*height,Wb,Wr);
	return output;
#else
	log[warning] << "CUDA optimizations not compiled in. Not doing anything" << std::endl;
	return output;
#endif
}

void YuriConvertor::generate_tables()
{
	boost::mutex::scoped_lock l(tables_lock);
	if (tables_initialized) return;
	// Generate table for YCx pairs ( Y | Cx<<8 )
/*	if (alternative_tables) {
		// outputs like abcd efgh ijAB CDEF GHIJ
		for (int i=0;i<256;++i) {
			sixteenbits_yc_b0[i] = (convY(i)&0x3FC)>>2;
			sixteenbits_yc_b2[i] = (convC(i)&0xF)<<4;
			sixteenbits_yc_b0b[i] = (convY(i)&0x3C0)>>6;
			sixteenbits_yc_b2b[i] = (convC(i)&0xFF);
		}
		for (int i=0;i<65536;++i) {
			sixteenbits_yc_b1[i] = (convY(i&0xFF)&0x3)<<6 | (convC((i>>8)&0xFF)&0x3F0)>>4;
			sixteenbits_yc_b1b[i] = (convY(i&0xFF)&0x3F)<<2 | (convC((i>>8)&0xFF)&0x300)>>8;
		}
	} else {*/
		// This results in output like cdef ghij EFGH IJab xxxx ABCD
		for (int i=0;i<256;++i) {
			sixteenbits_yc_b0[i] = (convY(i)&0xFF);
			sixteenbits_yc_b2[i] = (convC(i)&0x3C0)>>6;
			sixteenbits_yc_b0b[i] = (convY(i)&0xF)<<4;
			sixteenbits_yc_b2b[i] = (convC(i)&0x3FC)>>2;
		}
		for (int i=0;i<65536;++i) {
			sixteenbits_yc_b1[i] = (convY(i&0xFF)&0x300)>>8 | (convC((i>>8)&0xFF)&0x3F)<<2;
			sixteenbits_yc_b1b[i] = (convY(i&0xFF)&0x3F0)>>4 | (convC((i>>8)&0xFF)&0x3)<<6;
		}
	/*}*/

	log[info] << "Generating RGB->YCrCb conversion tables. This may take a while (128MB)" << std::endl;
// Generate tables for RGB24->YCrCb (20bit)
	yuri::ubyte_t R,G,B;
	yuri::ushort_t Y,Cr,Cb;
	float r,g,b, y, cb,cr;
	float Wg, Kb, Kr;
	Wg = 1.0f - Wr - Wb;
	Kb = 0.5f / (1.0f - Wb);
	Kr = 0.5f / (1.0f - Wr);

	for (yuri::uint_t i=0;i< (1<<24);++i) {
		B=(i>>16)&0xFF;
		G=(i>> 8)&0xFF;
		R= i     &0xFF;
		r = static_cast<float>(R)/256.0f;
		g = static_cast<float>(G)/256.0f;
		b = static_cast<float>(B)/256.0f;
		//
		y  = Wr * r + Wg * g + Wb * b;
		cb = (b - y) * Kb;
		cr = (r - y) * Kr;
		cb = std::max(-0.5f,std::min(cb,0.5f));
		cr = std::max(-0.5f,std::min(cr,0.5f));
		// Y should be between 64 - 940 (although 4-1020 is not violating anything)
		// Cb/Cr should be between 64 - 960 (although 4-1020 is not violating anything)
		Y  = static_cast<yuri::ushort_t>(64 + y  * 876);
		Cr = static_cast<yuri::ushort_t>(512 + cr * 896);
		Cb = static_cast<yuri::ushort_t>(512 + cb * 896);

		// Apparently I still need 'cdef ghij EFGH IJab xxxx ABCD'
		rgb_y0[i]=(Y>>0)&0xFF;  //'cdef ghij'
		rgb_y1[i]=(Y>>8)&0x03;  //'xxxx xxab'
		rgb_y2[i]=(Y<<4)&0xF0;  //'ghij xxxx'
		rgb_y3[i]=(Y>>4)&0x3F;  //'xxab cdef'
		rgb_cb[i]=Cb;
		rgb_cr[i]=Cr;

	}
	log[info] << "Tables generated" <<std::endl;
	tables_initialized = true;
}
template<> shared_ptr<BasicFrame> YuriConvertor::convert<YURI_FMT_V210_MVTP,YURI_FMT_YUV422,false>(shared_ptr<BasicFrame> frame)
{
	shared_ptr<BasicFrame> output;
	yuri::size_t w = frame->get_width();
	yuri::size_t h = frame->get_height();
	yuri::size_t pixel_pairs = (w*h)>>1;
	output=allocate_empty_frame(YURI_FMT_YUV422,w,h,true);
	yuri::ubyte_t *src = (*frame) [0].data.get();
	yuri::ubyte_t *dest= (*output)[0].data.get();
	yuri::ushort_t y1,y2,u,v;
	while(pixel_pairs--) {
		y1=*src++&0xFF;
		y1|=(*src&0x3)<<8;
		u=(*src++&0xFC)>>2;
		u|=(*src&0xF)<<6;
		y2=(*src++&0xF0)>>4;
		y2|=(*src&0x3F)<<4;
		v=(*src++&0xC0)>>6;
		v|=(*src++&0xFF)<<2;
		*dest++=(y1>>2)&0xFF;
		*dest++=(u>>2)&0xFF;
		*dest++=(y2>>2)&0xFF;
		*dest++=(v>>2)&0xFF;
	}
	return output;
}
bool YuriConvertor::set_param(Parameter &p)
{
	if (iequals(p.name,"colorimetry")) {
		std::string clr = p.get<std::string>();
		if (iequals(clr,"BT709") || iequals(clr,"REC709") || iequals(clr,"BT.709") || iequals(clr,"REC.709")) {
			colorimetry=YURI_COLORIMETRY_REC709;
		} else if (iequals(clr,"BT601") || iequals(clr,"REC601") || iequals(clr,"BT.601") || iequals(clr,"REC.601")) {
			colorimetry=YURI_COLORIMETRY_REC601;
		} else {
			log[warning] << "Unrecognized colorimetry type " << clr << ". Falling back to REC.709" << std::endl;
			colorimetry=YURI_COLORIMETRY_REC709;
		}
	} else if (iequals(p.name,"cuda")) {
			use_cuda = p.get<bool>();
			if (use_cuda) log[info] << "Using CUDA" << std::endl;
	} else if (iequals(p.name,"format")) {
		format = BasicPipe::get_format_from_string(p.get<std::string>(),YURI_TYPE_VIDEO);
		if (format==YURI_FMT_NONE) format=YURI_FMT_V210_MVTP;
	}
	else return BasicIOThread::set_param(p);
	return true;
}

#ifdef YURI_HAVE_CUDA
shared_ptr<void> YuriConvertor::cuda_alloc(yuri::size_t size)
{
	return shared_ptr<void> (CudaAlloc(size), &CudaDealloc);
}
#endif

}

}



