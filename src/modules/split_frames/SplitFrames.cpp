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

REGISTER("split_frames",SplitFrames)
IO_THREAD_GENERATOR(SplitFrames)

core::pParameters SplitFrames::configure()
{
	core::pParameters p = core::BasicIOThread::configure();
	p->set_description("Split frames from input pipe to several output pipes.");
	(*p)["outputs"]["Number of outputs"]=2;
	p->set_max_pipes(1,-1);
	return p;
}


SplitFrames::SplitFrames(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters):
core::BasicIOThread(log_,parent,1,1,std::string("split_frames")),outputs_(2),current_output_(0)
{
	IO_THREAD_INIT("SplitFrames")
	resize(1,outputs_);
}

SplitFrames::~SplitFrames()
{
}

bool SplitFrames::step()
{
	if (!in[0]) return true;
	core::pBasicFrame frame = in[0]->pop_frame();
	if (!frame) return true;

	push_raw_frame(current_output_,frame);
	current_output_=(current_output_+1)%outputs_;
	return true;
}
bool SplitFrames::set_param(const core::Parameter &param)
{
	if (param.name == "outputs") {
		outputs_ = param.get<size_t>();
	} else return core::BasicIOThread::set_param(param);
	return true;
}

} /* namespace split */
} /* namespace yuri */

