/*!
 * @file 		MultiIOFilter.cpp
 * @author 		Zdenek Travnicek
 * @date 		14.9.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "MultiIOFilter.h"
#include "yuri/core/utils/assign_parameters.h"
#include "yuri/core/utils/irange.h"
#include <cassert>

namespace yuri {
namespace core {

Parameters MultiIOFilter::configure()
{
	Parameters p = IOThread::configure();
//	p["realtime"]["Read always latest available, frame reducing latency, but dropping frames"]=false;
	p["main_input"]["Index of input that should trigger the processing. If specified, the processing will be invoked on each change of this input. Set to -1 to disable"]=-1;
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

std::vector<pFrame> MultiIOFilter::single_step(std::vector<pFrame> frames)
{
	return do_single_step(std::move(frames));
}
void MultiIOFilter::resize(position_t inp, position_t outp)
{
	stored_frames_.resize(inp);
	IOThread::resize(inp, outp);
}
bool MultiIOFilter::step()
{
	bool ready = true;
//	bool change = false;
	assert(get_no_in_ports()>0);
	assert(stored_frames_.size() == static_cast<size_t>(get_no_in_ports()));
	for (position_t i=0; i< get_no_in_ports(); ++i) {

		if (main_input_ == -2 || main_input_ == -3) {
			auto f = pop_frame(i);
			if (f) {
				stored_frames_[i]=f;
//				change = true;
			}
		} else if (!stored_frames_[i] || (main_input_>=0 && i!=main_input_)) {
				auto f = pop_frame(i);//]->pop_frame();
				if (f) stored_frames_[i] = f;
		}
		if (!stored_frames_[i]) ready = false;
	}
	if (ready || main_input_ == -3) {
//		log[log::info] << "use count before single_step(): " << stored_frames_[0].use_count();
		// In some cases, we will delete the frames (all or some) later.
		// So let's do it now and prepare new vector - so the implementation
		// can get unique copy
		std::vector<pFrame> frames_to_pass_down;
		const auto input_count = get_no_in_ports();
		if (main_input_ == -1 || main_input_ >= input_count || (input_count == 1 && main_input_ == 0)) {
			// We won't store any frames whatsoever
			frames_to_pass_down = std::move(stored_frames_);
			stored_frames_.resize(0);
		} else if (main_input_ >= 0) {
			// We have to store all frames except one (for main_input)
			frames_to_pass_down.resize(input_count, pFrame{});
			for (auto i: irange(0, input_count)) {
				if (i != main_input_) frames_to_pass_down[i] = stored_frames_[i];
				else {
					frames_to_pass_down[i]=std::move(stored_frames_[i]);
					stored_frames_[i].reset(); // This should not be necessary, but just to make sure...
				}
			}
		} else {
			// We have to keep all frames, so let's just copy the pointers;
			frames_to_pass_down = stored_frames_;
		}

		// make sure we didn't mess up the stored_frames_
		stored_frames_.resize(input_count, pFrame{});
		auto outframes = single_step(std::move(frames_to_pass_down));

		for (position_t i=0; i< std::min(get_no_out_ports(), static_cast<position_t>(outframes.size())); ++i) {
			if (outframes[i]) push_frame(i, outframes[i]);
		}

		if (main_input_ == -2 || main_input_ == -3) {
			// Nothing to do
		} else if (main_input_ < 0 || main_input_ >= get_no_in_ports()) {
			// If there's none or invalid main_input, we have to clean everything
			for (auto& sf: stored_frames_) sf.reset();
		} else { // Otherwise, let's clean just the main input
			stored_frames_[main_input_].reset();
		}
	}
	return true;
}

bool MultiIOFilter::set_param(const Parameter &parameter)
{
	if (assign_parameters(parameter)
			(main_input_, "main_input"))
		return true;
	return IOThread::set_param(parameter);
}

}
}
