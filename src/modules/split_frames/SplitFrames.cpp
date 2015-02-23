/*
 * SplitFrames.cpp
 *
 *  Created on: 23.2.2013
 *      Author: neneko
 */


#include "SplitFrames.h"
#include "yuri/core/Module.h"

namespace yuri {
namespace split {

IOTHREAD_GENERATOR(SplitFrames)

MODULE_REGISTRATION_BEGIN("split_frames")
		REGISTER_IOTHREAD("split_frames",SplitFrames)
MODULE_REGISTRATION_END()

core::Parameters SplitFrames::configure()
{
	core::Parameters p = core::MultiIOFilter::configure();
	p.set_description("Split frames from input pipe to several output pipes.");
	p["outputs"]["Number of outputs"]=2;
//	p->set_max_pipes(1,-1);
	return p;
}


SplitFrames::SplitFrames(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::MultiIOFilter(log_,parent,1,1,std::string("split_frames")),outputs_(2),current_output_(0)
{
	IOTHREAD_INIT(parameters)
	resize(1,outputs_);
}

SplitFrames::~SplitFrames() noexcept
{
}

std::vector<core::pFrame> SplitFrames::do_single_step(std::vector<core::pFrame> frames)
//bool SplitFrames::step()
{
//	if (!in[0]) return true;
//	core::pBasicFrame frame = in[0]->pop_frame();
//	if (!frame) return true;

	//push_frame(current_output_,frames[0]);
	std::vector<core::pFrame> outputs(outputs_);
	outputs[current_output_]=frames[0];
	current_output_=(current_output_+1)%outputs_;
	return outputs;
}
bool SplitFrames::set_param(const core::Parameter &param)
{
	if (param.get_name() == "outputs") {
		outputs_ = param.get<size_t>();
	} else return core::MultiIOFilter::set_param(param);
	return true;
}

} /* namespace split */
} /* namespace yuri */

