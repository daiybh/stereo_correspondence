/*
 * MultiIOFilter.cpp
 *
 *  Created on: 14.9.2013
 *      Author: neneko
 */

#include "MultiIOFilter.h"
#include <cassert>

namespace yuri {
namespace core {

Parameters MultiIOFilter::configure()
{
	Parameters p = IOThread::configure();
//	p["realtime"]["Read always latest available, frame reducing latency, but dropping frames"]=false;
	p["main_input"]["Index of input that should trigger the processing. If specified, the precessing will be invoked on each change of this input. Set to -1 to disable"]=-1;
	return p;
}

MultiIOFilter::MultiIOFilter(const log::Log &log_, pwThreadBase parent,
		position_t inp, position_t outp, const std::string& id)
:IOThread(log_, parent, inp, outp, id),stored_frames_(inp),//realtime_(false),
 main_input_(-1)
{
	set_latency(10_ms);
}

MultiIOFilter::~MultiIOFilter() noexcept
{

}

std::vector<pFrame> MultiIOFilter::single_step(const std::vector<pFrame>& frames)
{
	return do_single_step(frames);
}
void MultiIOFilter::resize(position_t inp, position_t outp)
{
	stored_frames_.resize(inp);
	IOThread::resize(inp, outp);
}
bool MultiIOFilter::step()
{
	bool ready = true;
	assert(get_no_in_ports()>0);
	assert(stored_frames_.size() == static_cast<size_t>(get_no_in_ports()));
	for (position_t i=0; i< get_no_in_ports(); ++i) {
//		if (realtime_) {
//			auto f = pop_frame(i);
//			if (f) stored_frames_[i] = f;
//		}
//		else {
			if (!stored_frames_[i] || (main_input_>=0 && i!=main_input_)) {
				auto f = pop_frame(i);//]->pop_frame();
				if (f) stored_frames_[i] = f;
			}
//		}
		if (!stored_frames_[i]) ready = false;
	}
	if (ready) {
		auto outframes = single_step(stored_frames_);
		for (position_t i=0; i< std::min(get_no_out_ports(), static_cast<position_t>(outframes.size())); ++i) {
			if (outframes[i]) push_frame(i, outframes[i]);
		}
		// If there's none or invalid main_input, we have to clean everything
		if (main_input_ < 0 || main_input_ >= get_no_in_ports()) {
			for (auto& sf: stored_frames_) sf.reset();
		} else { // Otherwise, let's clean just the main input
			stored_frames_[main_input_].reset();
		}
	}
	return true;
}

bool MultiIOFilter::set_param(const Parameter &parameter)
{
	/*if (iequals(parameter.name, "realtime")) {
		realtime_ = parameter.get<bool>();
	} else */if (iequals(parameter.get_name(), "main_input")) {
		main_input_ = parameter.get<position_t>();
	} else return IOThread::set_param(parameter);
	return true;
}

}
}
