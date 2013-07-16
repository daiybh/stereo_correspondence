/*
 * BasicIOFilter.cpp
 *
 *  Created on: 30.6.2013
 *      Author: neneko
 */

#include "BasicIOFilter.h"

namespace yuri {
namespace core {

pParameters BasicMultiIOFilter::configure()
{
	auto p = BasicIOThread::configure();
	(*p)["realtime"]["Read always latest available, frame reducing latency, but dropping frames"]=false;
	(*p)["main_input"]["Index of input that should trigger the processing. If specified, the precessing will be invoked on each change of thid input. Set to -1 to disable"]=-1;
	return p;
}

BasicMultiIOFilter::BasicMultiIOFilter(log::Log &log_, pwThreadBase parent,
					yuri::sint_t inp, yuri::sint_t outp, std::string id)
:BasicIOThread(log_, parent, inp, outp, id),stored_frames_(inp),realtime_(false),
 main_input_(-1)
{
}

BasicMultiIOFilter::~BasicMultiIOFilter()
{

}

std::vector<pBasicFrame> BasicMultiIOFilter::single_step(const std::vector<pBasicFrame>& frames)
{
	return do_single_step(frames);
}
bool BasicMultiIOFilter::step()
{
	bool ready = true;
	assert(in_ports>0);
	assert(stored_frames_.size() == static_cast<size_t>(in_ports));
	for (sint_t i=0; i< in_ports; ++i) {
		if (realtime_) {
			auto f = in[i]->pop_latest();
			if (f) stored_frames_[i] = f;
		}
		else {
			if (!stored_frames_[i]) {
				auto f = in[i]->pop_frame();
				if (f) stored_frames_[i] = f;
			}
		}
		if (!stored_frames_[i]) ready = false;
	}
	if (ready) {
		auto outframes = single_step(stored_frames_);
		for (size_t i=0; i< std::min(static_cast<size_t>(out_ports), outframes.size()); ++i) {
			if (outframes[i]) push_raw_frame(i, outframes[i]);
		}
		// If there's none or invalid main_input, we have to clean everything
		if (main_input_ < 0 || main_input_ >= in_ports) {
			for (auto& sf: stored_frames_) sf.reset();
		} else { // Otherwise, let's clean just the main input
			stored_frames_[main_input_].reset();
		}
	}
	return true;
}

bool BasicMultiIOFilter::set_param(const Parameter &parameter)
{
	if (iequals(parameter.name, "realtime")) {
		realtime_ = parameter.get<bool>();
	} else if (iequals(parameter.name, "main_input")) {
		main_input_ = parameter.get<ssize_t>();
	} else return BasicIOThread::set_param(parameter);
	return true;
}
BasicIOFilter::BasicIOFilter(log::Log &log_, pwThreadBase parent,std::string id)
:BasicMultiIOFilter(log_, parent, 1, 1, id) {

}

BasicIOFilter::~BasicIOFilter()
{

}

pBasicFrame	BasicIOFilter::simple_single_step(const pBasicFrame& frame)
{
	return do_simple_single_step(frame);
}

std::vector<pBasicFrame> BasicIOFilter::do_single_step(const std::vector<pBasicFrame>& frames)
{
	assert (frames.size() == 1 && frames[0]);
	const pBasicFrame& frame = frames[0];
	pBasicFrame outframe = simple_single_step(frame);
	if (outframe) return {outframe};
	return {};
}

}
}


