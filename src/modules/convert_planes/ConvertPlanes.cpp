/*!
 * @file 		ConvertPlanes.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		30.10.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "ConvertPlanes.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/raw_frame_params.h"
#include "yuri/core/thread/ConverterRegister.h"
namespace yuri {
namespace convert_planar {

namespace {

template<format_t in, format_t out, size_t planes>
core::pRawVideoFrame split_planes(core::pRawVideoFrame frame) {
	const resolution_t res = frame->get_resolution();
	core::pRawVideoFrame frame_out = core::RawVideoFrame::create_empty(out, res);
	typedef decltype(PLANE_DATA(frame_out,0).begin()) iter_t;
	std::vector<iter_t> iters(planes);
	iter_t iter_in = PLANE_DATA(frame, 0).begin();
//	iter_t iter_out = PLANE_DATA(frame_out, 0).begin();
	for (size_t i = 0; i < planes; ++i) {
		//iters[i]=iter_out+i;
		iters[i]=PLANE_DATA(frame_out, i).begin();
	}
	for (size_t line = 0; line < res.height; ++ line) {
		for (size_t col = 0; col < res.width; ++col) {
			for (size_t i = 0; i < planes; ++i) {
				*iters[i]++=*iter_in++;
			}
		}
	}
	return frame_out;
}

template<format_t in>
void store_yuv422(uint8_t*& it, uint8_t*& y, uint8_t*& u, uint8_t*& v);

template<>
void store_yuv422<core::raw_format::yuyv422>(uint8_t*& it, uint8_t*& y, uint8_t*& u, uint8_t*& v)
{
	*y++=*it++;
	*u++=*it++;
	*y++=*it++;
	*v++=*it++;
}

template<>
void store_yuv422<core::raw_format::uyvy422>(uint8_t*& it, uint8_t*& y, uint8_t*& u, uint8_t*& v)
{
	*u++=*it++;
	*y++=*it++;
	*v++=*it++;
	*y++=*it++;
}
template<>
void store_yuv422<core::raw_format::yvyu422>(uint8_t*& it, uint8_t*& y, uint8_t*& u, uint8_t*& v)
{
	*y++=*it++;
	*v++=*it++;
	*y++=*it++;
	*u++=*it++;
}

template<>
void store_yuv422<core::raw_format::vyuy422>(uint8_t*& it, uint8_t*& y, uint8_t*& u, uint8_t*& v)
{
	*v++=*it++;
	*y++=*it++;
	*u++=*it++;
	*y++=*it++;
}

template<format_t in, format_t out>
core::pRawVideoFrame split_planes_422p(core::pRawVideoFrame frame)
{
	const resolution_t res = frame->get_resolution();
	core::pRawVideoFrame frame_out = core::RawVideoFrame::create_empty(out, res);
	auto iter_in = PLANE_DATA(frame, 0).begin();
	auto iter_out0 = PLANE_DATA(frame_out, 0).begin();
	auto iter_out1 = PLANE_DATA(frame_out, 1).begin();
	auto iter_out2 = PLANE_DATA(frame_out, 2).begin();
	for (size_t line = 0; line < res.height; line+=1) {
		for (size_t col = 0; col < res.width; col+=2) {
			store_yuv422<in>(iter_in, iter_out0, iter_out1, iter_out2);
		}
	}
	return frame_out;
}



template<format_t format>
void store_yuv420(uint8_t*& it0, uint8_t*& it1, uint8_t*& y0, uint8_t*& y1, uint8_t*& u, uint8_t*& v);


template<>
void store_yuv420<core::raw_format::yuyv422>(uint8_t*& it0, uint8_t*& it1, uint8_t*& y0, uint8_t*& y1, uint8_t*& u, uint8_t*& v)
{
	uint_fast16_t va = 0;
	uint_fast16_t ua = 0;
	*y0++=*it0++;
	ua=*it0++;
	*y0++=*it0++;
	va=*it0++;

	*y1++=*it1++;
	ua+=*it1++;
	*y1++=*it1++;
	va+=*it1++;


	*v++=static_cast<uint8_t>(va/2);
	*u++=static_cast<uint8_t>(ua/2);
}
template<>
void store_yuv420<core::raw_format::yvyu422>(uint8_t*& it0, uint8_t*& it1, uint8_t*& y0, uint8_t*& y1, uint8_t*& u, uint8_t*& v)
{
	uint_fast16_t va = 0;
	uint_fast16_t ua = 0;
	*y0++=*it0++;
	va=*it0++;
	*y0++=*it0++;
	ua=*it0++;

	*y1++=*it1++;
	va+=*it1++;
	*y1++=*it1++;
	ua+=*it1++;

	*v++=static_cast<uint8_t>(va/2);
	*u++=static_cast<uint8_t>(ua/2);
}
template<>
void store_yuv420<core::raw_format::uyvy422>(uint8_t*& it0, uint8_t*& it1, uint8_t*& y0, uint8_t*& y1, uint8_t*& u, uint8_t*& v)
{
	uint_fast16_t va = 0;
	uint_fast16_t ua = 0;

	ua=*it0++;
	*y0++=*it0++;
	va=*it0++;
	*y0++=*it0++;

	ua+=*it1++;
	*y1++=*it1++;
	va+=*it1++;
	*y1++=*it1++;

	*v++=static_cast<uint8_t>(va/2);
	*u++=static_cast<uint8_t>(ua/2);
}
template<>
void store_yuv420<core::raw_format::vyuy422>(uint8_t*& it0, uint8_t*& it1, uint8_t*& y0, uint8_t*& y1, uint8_t*& u, uint8_t*& v)
{
	uint_fast16_t va = 0;
	uint_fast16_t ua = 0;

	va=*it0++;
	*y0++=*it0++;
	ua=*it0++;
	*y0++=*it0++;


	va+=*it1++;
	*y1++=*it1++;
	ua+=*it1++;
	*y1++=*it1++;

	*v++=static_cast<uint8_t>(va/2);
	*u++=static_cast<uint8_t>(ua/2);
}
template<format_t in, format_t out>
core::pRawVideoFrame split_planes_420p(core::pRawVideoFrame frame)
{
	const resolution_t res = frame->get_resolution();
	core::pRawVideoFrame frame_out = core::RawVideoFrame::create_empty(out, res);
	auto iter_out1 = PLANE_DATA(frame_out, 1).begin();
	auto iter_out2 = PLANE_DATA(frame_out, 2).begin();
	for (size_t line = 0; line < res.height; line+=2) {
		auto iter_in0 = PLANE_DATA(frame, 0).begin() + line*res.width*2;
		auto iter_in1 = PLANE_DATA(frame, 0).begin() + (line+1)*res.width*2;
		auto iter_out00 = PLANE_DATA(frame_out, 0).begin() + line*res.width;
		auto iter_out01 = PLANE_DATA(frame_out, 0).begin() + (line+1)*res.width;
		for (size_t col = 0; col < res.width; col+=2) {
			store_yuv420<in>(iter_in0, iter_in1, iter_out00, iter_out01, iter_out1, iter_out2);
		}
	}
	return frame_out;
}


template<format_t in>
void store_yuv411(uint8_t*& it, uint8_t*& y, uint8_t*& u, uint8_t*& v);

template<>
void store_yuv411<core::raw_format::yuyv422>(uint8_t*& it, uint8_t*& y, uint8_t*& u, uint8_t*& v)
{
	uint_fast16_t ua = 0;
	uint_fast16_t va = 0;
	*y++=*it++;
	ua=*it++;
	*y++=*it++;
	va=*it++;
	*y++=*it++;
	ua+=*it++;
	*y++=*it++;
	va+=*it++;

	*v++=static_cast<uint8_t>(va/2);
	*u++=static_cast<uint8_t>(ua/2);
}

template<>
void store_yuv411<core::raw_format::yvyu422>(uint8_t*& it, uint8_t*& y, uint8_t*& u, uint8_t*& v)
{
	uint_fast16_t ua = 0;
	uint_fast16_t va = 0;
	*y++=*it++;
	va=*it++;
	*y++=*it++;
	ua=*it++;
	*y++=*it++;
	va+=*it++;
	*y++=*it++;
	ua+=*it++;

	*v++=static_cast<uint8_t>(va/2);
	*u++=static_cast<uint8_t>(ua/2);
}

template<>
void store_yuv411<core::raw_format::uyvy422>(uint8_t*& it, uint8_t*& y, uint8_t*& u, uint8_t*& v)
{
	uint_fast16_t ua = 0;
	uint_fast16_t va = 0;

	ua=*it++;
	*y++=*it++;
	va=*it++;
	*y++=*it++;
	ua+=*it++;
	*y++=*it++;
	va+=*it++;
	*y++=*it++;

	*v++=static_cast<uint8_t>(va/2);
	*u++=static_cast<uint8_t>(ua/2);
}

template<>
void store_yuv411<core::raw_format::vyuy422>(uint8_t*& it, uint8_t*& y, uint8_t*& u, uint8_t*& v)
{
	uint_fast16_t ua = 0;
	uint_fast16_t va = 0;

	va=*it++;
	*y++=*it++;
	ua=*it++;
	*y++=*it++;
	va+=*it++;
	*y++=*it++;
	ua+=*it++;
	*y++=*it++;

	*v++=static_cast<uint8_t>(va/2);
	*u++=static_cast<uint8_t>(ua/2);
}
template<format_t in, format_t out>
core::pRawVideoFrame split_planes_411p(core::pRawVideoFrame frame)
{
	const resolution_t res = frame->get_resolution();
	core::pRawVideoFrame frame_out = core::RawVideoFrame::create_empty(out, res);
	auto iter_in = PLANE_DATA(frame, 0).begin();
	auto iter_out0 = PLANE_DATA(frame_out, 0).begin();
	auto iter_out1 = PLANE_DATA(frame_out, 1).begin();
	auto iter_out2 = PLANE_DATA(frame_out, 2).begin();
	for (size_t line = 0; line < res.height; line+=1) {
		for (size_t col = 0; col < res.width; col+=4) {
			store_yuv411<in>(iter_in, iter_out0, iter_out1, iter_out2);
		}
	}
	return frame_out;
}


template<format_t in, format_t out>
core::pRawVideoFrame merge_planes_sub3_xy(core::pRawVideoFrame frame) {
//	printf("BOO1\n");
	const resolution_t res = frame->get_resolution();
	core::pRawVideoFrame frame_out = core::RawVideoFrame::create_empty(out, res);
//	typedef decltype(PLANE_DATA(frame_out,0).begin()) iter_t;
//	std::vector<iter_t> iters(planes);
	auto iter_in0 = PLANE_DATA(frame, 0).begin();
	auto iter_in1 = PLANE_DATA(frame, 1).begin();
	auto iter_in2 = PLANE_DATA(frame, 2).begin();
	auto iter_out = PLANE_DATA(frame_out, 0).begin();
	for (size_t line = 0; line < res.height; line+=2) {
		auto it1 = iter_in1;
		auto it2 = iter_in2;
		for (size_t line2 = line; line2 < std::min(res.height,line+2); ++line2) {
			for (size_t col = 0; col < res.width; col+=2) {
				for (size_t col2 = col; col2 < std::min(res.width,col+2); ++col2) {
					*iter_out++ = *iter_in0++;
					*iter_out++ = *it1;
					*iter_out++ = *it2;
				}
				it1++;it2++;
			}
		}
		iter_in1++;iter_in2++;
	}
	return frame_out;
}
//template<format_t in, format_t out>
//core::pRawVideoFrame merge_planes_411p_422(core::pRawVideoFrame frame) {
//	const resolution_t res = frame->get_resolution();
//	core::pRawVideoFrame frame_out = core::RawVideoFrame::create_empty(out, res);
//	typedef decltype(PLANE_DATA(frame_out,0).begin()) iter_t;
//	iter_t iter_in0 = PLANE_DATA(frame, 0).begin();
//	iter_t iter_in1 = PLANE_DATA(frame, 1).begin();
//	iter_t iter_in2 = PLANE_DATA(frame, 2).begin();
//	iter_t iter_out = PLANE_DATA(frame_out, 0).begin();
//	for (size_t line = 0; line < res.height; ++line) {
//		for (size_t col = 0; col < res.width; col+=4) {
//				*iter_out++ = *iter_in0++;
//				*iter_out++ = *iter_in1;
//				*iter_out++ = *iter_in0++;
//				*iter_out++ = *iter_in2;
//				*iter_out++ = *iter_in0++;
//				*iter_out++ = *iter_in1++;
//				*iter_out++ = *iter_in0++;
//				*iter_out++ = *iter_in2++;
//		}
//	}
//	return frame_out;
//}
template<format_t in, format_t out>
core::pRawVideoFrame merge_planes_420p_yuyv(core::pRawVideoFrame frame) {

	const resolution_t res = frame->get_resolution();
	core::pRawVideoFrame frame_out = core::RawVideoFrame::create_empty(out, res);
	typedef decltype(PLANE_DATA(frame_out,0).begin()) iter_t;
	iter_t iter_in0 = PLANE_DATA(frame, 0).begin();
	iter_t iter_in1 = PLANE_DATA(frame, 1).begin();
	iter_t iter_in2 = PLANE_DATA(frame, 2).begin();
	iter_t iter_out = PLANE_DATA(frame_out, 0).begin();
	for (size_t line = 0; line < res.height; line+=2) {
		for (size_t line2 = line; line2 < std::min(res.height,line+2); ++line2) {
			iter_t it1 = iter_in1;
			iter_t it2 = iter_in2;
			for (size_t col = 0; col < res.width; col+=2) {
					*iter_out++ = *iter_in0++;
					*iter_out++ = *it1++;
					*iter_out++ = *iter_in0++;
					*iter_out++ = *it2++;
			}
		}
		iter_in1+=res.width/2;
		iter_in2+=res.width/2;
	}

	return frame_out;
}
template<format_t in, format_t out>
core::pRawVideoFrame merge_planes_420p_yvyu(core::pRawVideoFrame frame) {

	const resolution_t res = frame->get_resolution();
	core::pRawVideoFrame frame_out = core::RawVideoFrame::create_empty(out, res);
	typedef decltype(PLANE_DATA(frame_out,0).begin()) iter_t;
	iter_t iter_in0 = PLANE_DATA(frame, 0).begin();
	iter_t iter_in1 = PLANE_DATA(frame, 1).begin();
	iter_t iter_in2 = PLANE_DATA(frame, 2).begin();
	iter_t iter_out = PLANE_DATA(frame_out, 0).begin();
	for (size_t line = 0; line < res.height; line+=2) {
		for (size_t line2 = line; line2 < std::min(res.height,line+2); ++line2) {
			iter_t it1 = iter_in1;
			iter_t it2 = iter_in2;
			for (size_t col = 0; col < res.width; col+=2) {
					*iter_out++ = *iter_in0++;
					*iter_out++ = *it2++;
					*iter_out++ = *iter_in0++;
					*iter_out++ = *it1++;
			}
		}
		iter_in1+=res.width/2;
		iter_in2+=res.width/2;
	}

	return frame_out;
}
template<format_t in, format_t out>
core::pRawVideoFrame merge_planes_420p_uyvy(core::pRawVideoFrame frame) {

	const resolution_t res = frame->get_resolution();
	core::pRawVideoFrame frame_out = core::RawVideoFrame::create_empty(out, res);
	typedef decltype(PLANE_DATA(frame_out,0).begin()) iter_t;
	iter_t iter_in0 = PLANE_DATA(frame, 0).begin();
	iter_t iter_in1 = PLANE_DATA(frame, 1).begin();
	iter_t iter_in2 = PLANE_DATA(frame, 2).begin();
	iter_t iter_out = PLANE_DATA(frame_out, 0).begin();
	for (size_t line = 0; line < res.height; line+=2) {
		for (size_t line2 = line; line2 < std::min(res.height,line+2); ++line2) {
			iter_t it1 = iter_in1;
			iter_t it2 = iter_in2;
			for (size_t col = 0; col < res.width; col+=2) {
				*iter_out++ = *it1++;
				*iter_out++ = *iter_in0++;
				*iter_out++ = *it2++;
				*iter_out++ = *iter_in0++;
			}
		}
		iter_in1+=res.width/2;
		iter_in2+=res.width/2;
	}

	return frame_out;
}
template<format_t in, format_t out>
core::pRawVideoFrame merge_planes_420p_vyuy(core::pRawVideoFrame frame) {

	const resolution_t res = frame->get_resolution();
	core::pRawVideoFrame frame_out = core::RawVideoFrame::create_empty(out, res);
	typedef decltype(PLANE_DATA(frame_out,0).begin()) iter_t;
	iter_t iter_in0 = PLANE_DATA(frame, 0).begin();
	iter_t iter_in1 = PLANE_DATA(frame, 1).begin();
	iter_t iter_in2 = PLANE_DATA(frame, 2).begin();
	iter_t iter_out = PLANE_DATA(frame_out, 0).begin();
	for (size_t line = 0; line < res.height; line+=2) {
		for (size_t line2 = line; line2 < std::min(res.height,line+2); ++line2) {
			iter_t it1 = iter_in1;
			iter_t it2 = iter_in2;
			for (size_t col = 0; col < res.width; col+=2) {
				*iter_out++ = *it2++;
				*iter_out++ = *iter_in0++;
				*iter_out++ = *it1++;
				*iter_out++ = *iter_in0++;
			}
		}
		iter_in1+=res.width/2;
		iter_in2+=res.width/2;
	}

	return frame_out;
}

template<format_t fmt>
void store_yuv422_plane(uint8_t*& iter_y, uint8_t*& iter_u, uint8_t*& iter_v, uint8_t*& iter_yuv);

template<>
void store_yuv422_plane<core::raw_format::yuyv422>(uint8_t*& iter_y, uint8_t*& iter_u, uint8_t*& iter_v, uint8_t*& iter_yuv)
{
	*iter_yuv++ = *iter_y++;
	*iter_yuv++ = *iter_u++;
	*iter_yuv++ = *iter_y++;
	*iter_yuv++ = *iter_v++;
}

template<>
void store_yuv422_plane<core::raw_format::yvyu422>(uint8_t*& iter_y, uint8_t*& iter_u, uint8_t*& iter_v, uint8_t*& iter_yuv)
{
	*iter_yuv++ = *iter_y++;
	*iter_yuv++ = *iter_v++;
	*iter_yuv++ = *iter_y++;
	*iter_yuv++ = *iter_u++;
}

template<>
void store_yuv422_plane<core::raw_format::uyvy422>(uint8_t*& iter_y, uint8_t*& iter_u, uint8_t*& iter_v, uint8_t*& iter_yuv)
{
	*iter_yuv++ = *iter_u++;
	*iter_yuv++ = *iter_y++;
	*iter_yuv++ = *iter_v++;
	*iter_yuv++ = *iter_y++;
}

template<>
void store_yuv422_plane<core::raw_format::vyuy422>(uint8_t*& iter_y, uint8_t*& iter_u, uint8_t*& iter_v, uint8_t*& iter_yuv)
{
	*iter_yuv++ = *iter_v++;
	*iter_yuv++ = *iter_y++;
	*iter_yuv++ = *iter_u++;
	*iter_yuv++ = *iter_y++;
}


template<format_t in, format_t out>
core::pRawVideoFrame merge_planes_422p_yuyv(core::pRawVideoFrame frame) {

	const resolution_t res = frame->get_resolution();
	core::pRawVideoFrame frame_out = core::RawVideoFrame::create_empty(out, res);
	typedef decltype(PLANE_DATA(frame_out,0).begin()) iter_t;
	iter_t iter_in0 = PLANE_DATA(frame, 0).begin();
	iter_t iter_in1 = PLANE_DATA(frame, 1).begin();
	iter_t iter_in2 = PLANE_DATA(frame, 2).begin();
	iter_t iter_out = PLANE_DATA(frame_out, 0).begin();
	for (size_t line = 0; line < res.height; ++line) {
			for (size_t col = 0; col < res.width; col+=2) {
				store_yuv422_plane<out>(iter_in0, iter_in1, iter_in2, iter_out);
			}
	}
	return frame_out;
}


core::pFrame dispatch(core::pRawVideoFrame frame, format_t target) {
	if (!frame) return {};
	format_t source = frame->get_format();
	using namespace yuri::core::raw_format;
	core::pRawVideoFrame frame_out;
//	printf("BOO0\n");
	if (source == rgb24 && target == rgb24p) frame_out = split_planes<rgb24, rgb24p, 3>(frame);
	if (source == rgba32 && target == rgba32p) frame_out =  split_planes<rgba32, rgba32p, 4>(frame);
	if (source == yuv444 && target == yuv444p) frame_out =  split_planes<yuv444, yuv444p, 3>(frame);
	if (source == yuyv422 && target == yuv422p) frame_out =  split_planes_422p<yuyv422, yuv422p>(frame);
	if (source == uyvy422 && target == yuv422p) frame_out =  split_planes_422p<uyvy422, yuv422p>(frame);
	if (source == yvyu422 && target == yuv422p) frame_out =  split_planes_422p<yvyu422, yuv422p>(frame);
	if (source == vyuy422 && target == yuv422p) frame_out =  split_planes_422p<vyuy422, yuv422p>(frame);

	if (source == yuyv422 && target == yuv420p) frame_out =  split_planes_420p<yuyv422, yuv420p>(frame);
	if (source == yvyu422 && target == yuv420p) frame_out =  split_planes_420p<yvyu422, yuv420p>(frame);
	if (source == uyvy422 && target == yuv420p) frame_out =  split_planes_420p<uyvy422, yuv420p>(frame);
	if (source == vyuy422 && target == yuv420p) frame_out =  split_planes_420p<vyuy422, yuv420p>(frame);

	if (source == yuyv422 && target == yuv411p) frame_out =  split_planes_411p<yuyv422, yuv411p>(frame);
	if (source == yvyu422 && target == yuv411p) frame_out =  split_planes_411p<yvyu422, yuv411p>(frame);
	if (source == uyvy422 && target == yuv411p) frame_out =  split_planes_411p<uyvy422, yuv411p>(frame);
	if (source == vyuy422 && target == yuv411p) frame_out =  split_planes_411p<vyuy422, yuv411p>(frame);

	//	if (source == yuv420p && target == yuv444) frame_out =  merge_planes_sub3_xy<yuv420p, yuv444>(frame);
//	if (source == yuv411p && target == yuyv422) frame_out =  merge_planes_411p_422<yuv420p, yuyv422>(frame);
	if (source == yuv420p && target == yuyv422) frame_out =  merge_planes_420p_yuyv<yuv420p, yuyv422>(frame);
	if (source == yuv420p && target == yvyu422) frame_out =  merge_planes_420p_yvyu<yuv420p, yvyu422>(frame);
	if (source == yuv420p && target == uyvy422) frame_out =  merge_planes_420p_uyvy<yuv420p, uyvy422>(frame);
	if (source == yuv420p && target == vyuy422) frame_out =  merge_planes_420p_vyuy<yuv420p, vyuy422>(frame);

	if (source == yuv422p && target == yuyv422) frame_out =  merge_planes_422p_yuyv<yuv422p, yuyv422>(frame);
	if (source == yuv422p && target == yvyu422) frame_out =  merge_planes_422p_yuyv<yuv422p, yvyu422>(frame);
	if (source == yuv422p && target == uyvy422) frame_out =  merge_planes_422p_yuyv<yuv422p, uyvy422>(frame);
	if (source == yuv422p && target == vyuy422) frame_out =  merge_planes_422p_yuyv<yuv422p, vyuy422>(frame);
	if (frame_out) {
		frame_out->set_duration(frame->get_duration());
		frame_out->set_timestamp(frame->get_timestamp());
	}
	return frame_out;
}



}
IOTHREAD_GENERATOR(ConvertPlanes)

MODULE_REGISTRATION_BEGIN("convert_planar")
		REGISTER_IOTHREAD("convert_planar",ConvertPlanes)
//		REGISTER_CONVERTER(yuri::core::raw_format::rgb24, yuri::core::raw_format::rgb24p, "convert_planar", 5)
//		REGISTER_CONVERTER(yuri::core::raw_format::rgba32, yuri::core::raw_format::rgba32p, "convert_planar", 5)
		REGISTER_CONVERTER(yuri::core::raw_format::yuv444, yuri::core::raw_format::yuv444p, "convert_planar", 5)
//		REGISTER_CONVERTER(yuri::core::raw_format::yuv420p, yuri::core::raw_format::yuv444, "convert_planar", 5)
		REGISTER_CONVERTER(yuri::core::raw_format::yuyv422, yuri::core::raw_format::yuv422p, "convert_planar", 5)
		REGISTER_CONVERTER(yuri::core::raw_format::uyvy422, yuri::core::raw_format::yuv422p, "convert_planar", 5)
		REGISTER_CONVERTER(yuri::core::raw_format::yvyu422, yuri::core::raw_format::yuv422p, "convert_planar", 5)
		REGISTER_CONVERTER(yuri::core::raw_format::vyuy422, yuri::core::raw_format::yuv422p, "convert_planar", 5)

		REGISTER_CONVERTER(yuri::core::raw_format::yuyv422, yuri::core::raw_format::yuv420p, "convert_planar", 6)
		REGISTER_CONVERTER(yuri::core::raw_format::yvyu422, yuri::core::raw_format::yuv420p, "convert_planar", 6)
		REGISTER_CONVERTER(yuri::core::raw_format::uyvy422, yuri::core::raw_format::yuv420p, "convert_planar", 6)
		REGISTER_CONVERTER(yuri::core::raw_format::vyuy422, yuri::core::raw_format::yuv420p, "convert_planar", 6)

		REGISTER_CONVERTER(yuri::core::raw_format::yuyv422, yuri::core::raw_format::yuv411p, "convert_planar", 6)
		REGISTER_CONVERTER(yuri::core::raw_format::yvyu422, yuri::core::raw_format::yuv411p, "convert_planar", 6)
		REGISTER_CONVERTER(yuri::core::raw_format::uyvy422, yuri::core::raw_format::yuv411p, "convert_planar", 6)
		REGISTER_CONVERTER(yuri::core::raw_format::vyuy422, yuri::core::raw_format::yuv411p, "convert_planar", 6)


		REGISTER_CONVERTER(yuri::core::raw_format::yuv420p, yuri::core::raw_format::yuyv422, "convert_planar", 5)
		REGISTER_CONVERTER(yuri::core::raw_format::yuv420p, yuri::core::raw_format::yvyu422, "convert_planar", 5)
		REGISTER_CONVERTER(yuri::core::raw_format::yuv420p, yuri::core::raw_format::uyvy422, "convert_planar", 5)
		REGISTER_CONVERTER(yuri::core::raw_format::yuv420p, yuri::core::raw_format::vyuy422, "convert_planar", 5)

		REGISTER_CONVERTER(yuri::core::raw_format::yuv422p, yuri::core::raw_format::yuyv422, "convert_planar", 5)
		REGISTER_CONVERTER(yuri::core::raw_format::yuv422p, yuri::core::raw_format::yvyu422, "convert_planar", 5)
		REGISTER_CONVERTER(yuri::core::raw_format::yuv422p, yuri::core::raw_format::uyvy422, "convert_planar", 5)
		REGISTER_CONVERTER(yuri::core::raw_format::yuv422p, yuri::core::raw_format::vyuy422, "convert_planar", 5)

MODULE_REGISTRATION_END()

core::Parameters ConvertPlanes::configure()
{
	core::Parameters p = core::SpecializedIOFilter<core::RawVideoFrame>::configure();
	p.set_description("ConvertPlanes");
	p["format"]["Target format"]="YUV";
	return p;
}


ConvertPlanes::ConvertPlanes(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::SpecializedIOFilter<core::RawVideoFrame>(log_,parent,std::string("convert_planar"))
{
	IOTHREAD_INIT(parameters)
}

ConvertPlanes::~ConvertPlanes() noexcept
{
}

core::pFrame ConvertPlanes::do_special_single_step(core::pRawVideoFrame frame)
{
	return dispatch(frame, format_);
}

core::pFrame ConvertPlanes::do_convert_frame(core::pFrame input_frame, format_t target_format)
{
	core::pRawVideoFrame frame = dynamic_pointer_cast<core::RawVideoFrame>(input_frame);
	if (!frame) {
		log[log::warning] << "Got bad frame type!!";
		return {};
	}
	return dispatch(frame, target_format);
}
bool ConvertPlanes::set_param(const core::Parameter& param)
{
	if (param.get_name() == "format") {
		format_ = core::raw_format::parse_format(param.get<std::string>());
	}
	return core::SpecializedIOFilter<core::RawVideoFrame>::set_param(param);
}

} /* namespace convert_planar */
} /* namespace yuri */

