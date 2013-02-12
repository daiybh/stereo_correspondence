/*
 * YuriConvertor.h
 *
 *  Created on: Aug 13, 2010
 *      Author: neneko
 */

#ifndef YURICONVERTOR_H_
#define YURICONVERTOR_H_

#include "yuri/io/BasicIOThread.h"
#include "yuri/config/RegisteredClass.h"



namespace yuri {

namespace video {
using namespace yuri::io;
using namespace yuri::log;
using namespace yuri::config;

enum _colorimetry {
	YURI_COLORIMETRY_REC709,
	YURI_COLORIMETRY_REC601
};

class YuriConvertor: public BasicIOThread {
public:
	YuriConvertor(Log &log_, pThreadBase parent,Parameters& parameters) IO_THREAD_CONSTRUCTOR;
	virtual ~YuriConvertor();
	IO_THREAD_GENERATOR_DECLARATION
	static shared_ptr<Parameters> configure();
	bool step();
	bool set_param(Parameter &p);
protected:
	template<yuri::format_t format_in,yuri::format_t format_out,bool using_cuda> shared_ptr<BasicFrame> convert(shared_ptr<BasicFrame> frame);

	//shared_ptr<BasicFrame> convert_from_yuv422(shared_ptr<BasicFrame> frame);
//	shared_ptr<BasicFrame> convert_rgb_to_ycbcr(shared_ptr<BasicFrame> frame);
//	shared_ptr<BasicFrame> convert_rgb_to_ycbcr_cuda(shared_ptr<BasicFrame> frame);
//	shared_ptr<BasicFrame> convert_yuv_to_rgb_cuda(shared_ptr<BasicFrame> frame);
//	shared_ptr<BasicFrame> convert_mvtp_to_yuv(shared_ptr<BasicFrame> frame);

	bool use_cuda;

	void generate_tables();

	_colorimetry colorimetry;
	float Wb, Wr;
	shared_ptr<void> cuda_src, cuda_dest;
	yuri::size_t allocated_size;
	yuri::format_t format;
	static bool tables_initialized;
	static yuri::ubyte_t sixteenbits_yc_b0[256];
	static yuri::ubyte_t sixteenbits_yc_b1[65536];
	static yuri::ubyte_t sixteenbits_yc_b2[256];
	static yuri::ubyte_t sixteenbits_yc_b0b[256];
	static yuri::ubyte_t sixteenbits_yc_b1b[65536];
	static yuri::ubyte_t sixteenbits_yc_b2b[256];

	// 128MB of ancillary tables for RGB24->YCbCr
	static yuri::ubyte_t rgb_y0[1<<24];
	static yuri::ubyte_t rgb_y1[1<<24];
	static yuri::ubyte_t rgb_y2[1<<24];
	static yuri::ubyte_t rgb_y3[1<<24];
	static yuri::ushort_t rgb_cr[1<<24];
	static yuri::ushort_t rgb_cb[1<<24];

	static boost::mutex tables_lock;

#ifdef YURI_HAVE_CUDA
	static shared_ptr<void> cuda_alloc(yuri::size_t size);
#endif
	typedef shared_ptr<BasicFrame>(YuriConvertor::*converter_t)(shared_ptr<BasicFrame>);
	static std::map<std::pair<yuri::format_t, yuri::format_t>, converter_t> converters;
#ifdef YURI_HAVE_CUDA
	static std::map<std::pair<yuri::format_t, yuri::format_t>, converter_t> cuda_converters;
#endif

	//bool alternative_tables ;
public:
	static inline unsigned int convY(unsigned int Y) { return (Y*219+4128) >> 6; }
	static inline unsigned int convC(unsigned int C) {	return (C*7+129) >> 1; }

};

}

}

#endif /* YURICONVERTOR_H_ */
