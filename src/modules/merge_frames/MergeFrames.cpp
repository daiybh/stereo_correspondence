/*
 * MergeFrames.cpp
 *
 *  Created on: 23.2.2013
 *      Author: neneko
 */


#include "MergeFrames.h"
#include "yuri/core/Module.h"

namespace yuri {
namespace split {

IOTHREAD_GENERATOR(MergeFrames)

MODULE_REGISTRATION_BEGIN("merge_frames")
		REGISTER_IOTHREAD("merge_frames",MergeFrames)
MODULE_REGISTRATION_END()

core::Parameters MergeFrames::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("Merge frames from input pipes to single output pipe.");
	p["inputs"]["Number of inputs"]=2;
//	p->set_max_pipes(-1,1);
	return p;
}


MergeFrames::MergeFrames(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,1,1,std::string("merge_frames")),inputs_(2),current_input_(0)
{
	IOTHREAD_INIT(parameters)
	resize(inputs_,1);
}

MergeFrames::~MergeFrames() noexcept
{
}

bool MergeFrames::step()
{
	core::pFrame frame = pop_frame(current_input_);
	if (!frame) return true;
	push_frame(0,frame);
	current_input_=(current_input_+1)%inputs_;
	return true;
}
bool MergeFrames::set_param(const core::Parameter &param)
{
	if (param.get_name() == "inputs") {
		inputs_ = param.get<size_t>();
	} else return core::IOThread::set_param(param);
	return true;
}

} /* namespace split */
} /* namespace yuri */

