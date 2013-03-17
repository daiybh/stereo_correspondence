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

REGISTER("merge_frames",MergeFrames)
IO_THREAD_GENERATOR(MergeFrames)

core::pParameters MergeFrames::configure()
{
	core::pParameters p = core::BasicIOThread::configure();
	p->set_description("Merge frames from input pipes to single output pipe.");
	(*p)["inputs"]["Number of inputs"]=2;
	p->set_max_pipes(-1,1);
	return p;
}


MergeFrames::MergeFrames(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters):
core::BasicIOThread(log_,parent,1,1,std::string("merge_frames")),inputs_(2),current_input_(0)
{
	IO_THREAD_INIT("MergeFrames")
	resize(inputs_,1);
}

MergeFrames::~MergeFrames()
{
}

bool MergeFrames::step()
{
	if (!in[current_input_]) return true;
	core::pBasicFrame frame = in[current_input_]->pop_frame();
	if (!frame) return true;

	push_raw_frame(0,frame);
	current_input_=(current_input_+1)%inputs_;
	return true;
}
bool MergeFrames::set_param(const core::Parameter &param)
{
	if (param.name == "inputs") {
		inputs_ = param.get<size_t>();
	} else return core::BasicIOThread::set_param(param);
	return true;
}

} /* namespace split */
} /* namespace yuri */

