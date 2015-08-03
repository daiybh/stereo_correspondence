/*!
 * @file 		FrameInfo.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		28.10.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "FrameInfo.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/frame/raw_frame_params.h"
#include "yuri/core/frame/CompressedVideoFrame.h"
#include "yuri/core/frame/compressed_frame_params.h"
#include "yuri/core/frame/RawAudioFrame.h"
#include "yuri/core/frame/raw_audio_frame_params.h"
#include "yuri/core/frame/EventFrame.h"
#include "yuri/core/utils/global_time.h"
namespace yuri {
namespace frame_info {


IOTHREAD_GENERATOR(FrameInfo)

MODULE_REGISTRATION_BEGIN("frame_info")
		REGISTER_IOTHREAD("frame_info",FrameInfo)
MODULE_REGISTRATION_END()

core::Parameters FrameInfo::configure()
{
	core::Parameters p = core::IOFilter::configure();
	p.set_description("FrameInfo");
	p["print_all"]["Print info about every frame. If set to false, only frames after change will be printed"]=false;
	p["print_time"]["Print timestamps of the siplayed frames"]=false;
	return p;
}

namespace {
std::string get_interlace_info(const core::pVideoFrame& frame)
{
	switch (frame->get_interlacing()) {
		case interlace_t::progressive: return "progressive";
		case interlace_t::segmented_frame: return "segmented frame";
		case interlace_t::splitted: return "splitted frame";
		case interlace_t::interlaced: {
			switch (frame->get_field_order()) {
				case field_order_t::top_field_first: return "interlaced, top-field-first";
				case field_order_t::bottom_field_first: return "interlaced, bottom-field-first";
				default:
					return "interlaced, unknown field order";
			}
		} break;
	}
	return "Unknown interlacing info";
}

void print_video_frame(log::Log& log, std::string fname, const core::pVideoFrame& frame)
{
	log[log::info] << "Frame with format '" << fname << "', resolution " << frame->get_resolution() << ", " << get_interlace_info(frame);

}

void print_frame(log::Log& log, core::pRawVideoFrame frame)
{
	const std::string& fname = core::raw_format::get_format_name(frame->get_format());
	print_video_frame(log, fname, frame);
}
void print_frame(log::Log& log, core::pCompressedVideoFrame frame)
{
	const std::string& fname = core::compressed_frame::get_format_name(frame->get_format());
	print_video_frame(log, fname, frame);
}
void print_frame(log::Log& log, core::pRawAudioFrame frame)
{
	//const auto& fi = core::raw_format::get_format_info(frame->get_format());
	const std::string& fname = core::raw_audio_format::get_format_name(frame->get_format());
	log[log::info] << "Frame with format '" << fname << "', sampling rate " << frame->get_sampling_frequency()
					<< "Hz, " << frame->get_channel_count() << " channels";
}
void print_frame(log::Log& log, core::pEventFrame frame)
{

	log[log::info] << "Event Frame with name '" << frame->get_name()
			<< "' and value '" << event::lex_cast_value<std::string>(frame->get_event()) << "'";
}

void print_frame(log::Log& log, core::pFrame frame) {
	if (auto f = std::dynamic_pointer_cast<core::RawVideoFrame>(frame)) {
		print_frame(log, f);
	} else if (auto f2 = std::dynamic_pointer_cast<core::CompressedVideoFrame>(frame)) {
		print_frame(log, f2);
	} else if (auto f3 = std::dynamic_pointer_cast<core::RawAudioFrame>(frame)) {
		print_frame(log, f3);
	} else if (auto f4 = std::dynamic_pointer_cast<core::EventFrame>(frame)) {
		print_frame(log, f4);
	} else {
		log[log::info] << "Unknown format (" << frame->get_format() << ")";
	}
}

bool same_format(const core::pRawVideoFrame& a, const core::pRawVideoFrame& b)
{
	return (a->get_format() == b->get_format()) &&
			(a->get_resolution() == b->get_resolution());
}
bool same_format(const core::pCompressedVideoFrame& a, const core::pCompressedVideoFrame& b)
{
	return (a->get_format() == b->get_format()) &&
			(a->get_resolution() == b->get_resolution());
}
bool same_format(const core::pRawAudioFrame& a, const core::pRawAudioFrame& b)
{
	return (a->get_format() == b->get_format()) &&
			(a->get_sampling_frequency() == b->get_sampling_frequency()) &&
			(a->get_channel_count() == b->get_channel_count());
}

bool same_format(const core::pFrame& a, const core::pFrame& b)
{
	if (a->get_format() != b->get_format()) return false;
	{
		auto fa = std::dynamic_pointer_cast<core::RawVideoFrame>(a);
		auto fb = std::dynamic_pointer_cast<core::RawVideoFrame>(b);
		if (fa && fb) return same_format(fa, fb);
	}
	{
		auto fa = std::dynamic_pointer_cast<core::CompressedVideoFrame>(a);
		auto fb = std::dynamic_pointer_cast<core::CompressedVideoFrame>(b);
		if (fa && fb) return same_format(fa, fb);
	}
	{
		auto fa = std::dynamic_pointer_cast<core::RawAudioFrame>(a);
		auto fb = std::dynamic_pointer_cast<core::RawAudioFrame>(b);
		if (fa && fb) return same_format(fa, fb);
	}
	{
		auto fa = std::dynamic_pointer_cast<core::EventFrame>(a);
		auto fb = std::dynamic_pointer_cast<core::EventFrame>(b);
		if (fa && fb) return false;
	}
	return true;
}


}


FrameInfo::FrameInfo(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOFilter(log_,parent,std::string("frame_info")),
print_all_(false)
{
	IOTHREAD_INIT(parameters)
}

FrameInfo::~FrameInfo() noexcept
{
}

core::pFrame FrameInfo::do_simple_single_step(core::pFrame frame)
{
	try {
		if (print_all_ || !last_frame_ || !same_format(last_frame_, frame)) {
			print_frame(log, frame);
			if (print_time_) {
				log[log::info] << "\tTimestamp: " << (frame->get_timestamp() - core::utils::get_global_start_time());
			}

		}
	}
	catch (std::exception&){}
	last_frame_ = frame;
	return frame;
}
bool FrameInfo::set_param(const core::Parameter& param)
{
	if(assign_parameters(param)
			(print_all_, "print_all")
			(print_time_, "print_time"))
		return true;
	return core::IOFilter::set_param(param);
}

} /* namespace frame_info */
} /* namespace yuri */
