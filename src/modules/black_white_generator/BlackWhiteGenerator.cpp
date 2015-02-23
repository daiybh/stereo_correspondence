/*!
 * @file 		BlackWhiteGenerator.cpp
 * @author 		<Your name>
 * @date		13.05.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#include "BlackWhiteGenerator.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/raw_frame_params.h"
#include "yuri/core/frame/RawVideoFrame.h"
namespace yuri {
namespace black_white_generator {


IOTHREAD_GENERATOR(BlackWhiteGenerator)

MODULE_REGISTRATION_BEGIN("black_white_generator")
		REGISTER_IOTHREAD("black_white_generator",BlackWhiteGenerator)
MODULE_REGISTRATION_END()

core::Parameters BlackWhiteGenerator::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("BlackWhiteGenerator");
	p["frequency"]["Frequency of black.white switching."]=10;
	p["resolution"]["Resolution"]=resolution_t{800,600};
	p["format"]["Output format"]="yuv";
	return p;
}

namespace {

template<uint8_t value, uint8_t value2 = 128>
void fill_frame_yuv(core::pRawVideoFrame& frame) {
	auto& plane = PLANE_DATA(frame,0);
	const auto res = plane.get_resolution();
	auto iter = plane.begin();
	for (dimension_t line = 0; line < res.height; ++ line) {
		for (dimension_t col = 0; col < res.width; ++col) {
			*iter++ = value;
			*iter++ = value2;
		}
	}
}
template<uint8_t value, uint8_t value2 = 128>
void fill_frame_yuv444(core::pRawVideoFrame& frame) {
	auto& plane = PLANE_DATA(frame,0);
	const auto res = plane.get_resolution();
	auto iter = plane.begin();
	for (dimension_t line = 0; line < res.height; ++ line) {
		for (dimension_t col = 0; col < res.width; ++col) {
			*iter++ = value;
			*iter++ = value2;
			*iter++ = value2;
		}
	}
}

core::pFrame prepare_black_frame(format_t format, resolution_t resolution)
{
	auto res = core::RawVideoFrame::create_empty(format, resolution);
	switch (format) {
		case core::raw_format::yuyv422:
		case core::raw_format::yvyu422:
			fill_frame_yuv<0,128>(res);
			break;
		case core::raw_format::uyvy422:
		case core::raw_format::vyuy422:
			fill_frame_yuv<128,0>(res);
			break;
		case core::raw_format::yuv444:
			fill_frame_yuv444<0,128>(res);
			break;
		default:
			for (auto& plane: *res) {
				std::fill(plane.begin(), plane.end(), 0);
			}; break;
	}

	return res;
}
core::pFrame prepare_white_frame(format_t format, resolution_t resolution)
{
	auto res = core::RawVideoFrame::create_empty(format, resolution);
	switch (format) {
		case core::raw_format::yuyv422:
		case core::raw_format::yvyu422:
			fill_frame_yuv<255,128>(res);
			break;
		case core::raw_format::uyvy422:
		case core::raw_format::vyuy422:
			fill_frame_yuv<128,255>(res);
			break;
		case core::raw_format::yuv444:
			fill_frame_yuv444<255,128>(res);
			break;
		default:
			for (auto& plane: *res) {
				std::fill(plane.begin(), plane.end(), 255);
			}; break;
	}

	return res;
}
}

BlackWhiteGenerator::BlackWhiteGenerator(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,1,1,std::string("black_white_generator")),duration_(40_ms),format_(core::raw_format::yuyv422),resolution_(resolution_t{1920,1080})
{
	IOTHREAD_INIT(parameters)
	if (!format_) throw exception::InitializationFailed("Output format not specified or unsupported");
	black_frame_ = prepare_black_frame(format_, resolution_);
	white_frame_ = prepare_white_frame(format_, resolution_);
	if (!black_frame_ || !white_frame_) throw exception::InitializationFailed("Failed to prepare output frames");
}

BlackWhiteGenerator::~BlackWhiteGenerator() noexcept
{
}

void BlackWhiteGenerator::run()
{
	start_time_ = timestamp_t();
	timestamp_t next_time = start_time_;
	bool white = false;
	while(still_running()) {
		timestamp_t cur_time = timestamp_t();
		if (cur_time >= next_time) {
			push_frame(0, white?white_frame_:black_frame_);
			white = !white;
			next_time+=duration_;
		} else {
			sleep((next_time - cur_time)/2);
		}
	}
	close_pipes();

}
bool BlackWhiteGenerator::set_param(const core::Parameter& param)
{
	if (assign_parameters(param)
			(format_, "format", [](const core::Parameter& p){ return core::raw_format::parse_format(p.get<std::string>());})
			(resolution_, "resolution")
			(duration_, "frequency", [](const core::Parameter& p) { return 1_s/p.get<double>(); }))
		return true;
	return core::IOThread::set_param(param);
}

} /* namespace black_white_generator */
} /* namespace yuri */
