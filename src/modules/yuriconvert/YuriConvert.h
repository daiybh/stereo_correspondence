/*!
 * @file 		YuriConvertor.h
 * @author 		Zdenek Travnicek
 * @date 		13.8.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef YURICONVERTOR_H_
#define YURICONVERTOR_H_

#include "yuri/core/BasicIOThread.h"

namespace yuri {

namespace video {

enum _colorimetry {
	YURI_COLORIMETRY_REC709,
	YURI_COLORIMETRY_REC601
};

class YuriConvertor: public core::BasicIOThread {
public:
	YuriConvertor(log::Log &log_, core::pwThreadBase parent, core::Parameters& parameters) IO_THREAD_CONSTRUCTOR;
	virtual ~YuriConvertor();
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();
	bool step();
	bool set_param(const core::Parameter &p);
protected:
	template<yuri::format_t format_in,yuri::format_t format_out,bool using_cuda> core::pBasicFrame convert(core::pBasicFrame frame);

	//core::pBasicFrame convert_from_yuv422(core::pBasicFrame frame);
//	core::pBasicFrame convert_rgb_to_ycbcr(core::pBasicFrame frame);
//	core::pBasicFrame convert_rgb_to_ycbcr_cuda(core::pBasicFrame frame);
//	core::pBasicFrame convert_yuv_to_rgb_cuda(core::pBasicFrame frame);
//	core::pBasicFrame convert_mvtp_to_yuv(core::pBasicFrame frame);

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

	static yuri::ushort_t tenbits_reverse[1<<10];
	static boost::mutex tables_lock;

#ifdef YURI_HAVE_CUDA
	static shared_ptr<void> cuda_alloc(yuri::size_t size);
#endif
	typedef core::pBasicFrame(YuriConvertor::*converter_t)(core::pBasicFrame);
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
