/*!
 * @file 		DummyModule.cpp
 * @author 		Zdenek Travnicek
 * @date 		11.2.2013
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "DummyModule.h"
#include "yuri/core/Module.h"

namespace yuri {
namespace dummy_module {

REGISTER("dummy",DummyModule)

IO_THREAD_GENERATOR(DummyModule)

// So we can write log[info] instead of log[log::info]
using namespace yuri::log;

core::pParameters DummyModule::configure()
{
	core::pParameters p = core::IOThread::configure();
	p->set_description("Dummy module. For testing only.");
	(*p)["size"]["Set size of ....  (ignored ;)"]=666;
	(*p)["name"]["Set name"]=std::string("");
	p->set_max_pipes(1,1);
	return p;
}


DummyModule::DummyModule(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters):
core::IOThread(log_,parent,1,1,std::string("dummy"))
{
	IO_THREAD_INIT("Dummy")
	if (!dummy_name.empty()) log[info] << "Got name " << dummy_name <<"\n";
}

DummyModule::~DummyModule()
{
}

bool DummyModule::step()
{
	core::pBasicFrame frame = in[0]->pop_frame();
	if (frame) {
		yuri::format_t fmt = frame->get_format();
		if (core::BasicPipe::get_format_group(fmt)==YURI_TYPE_VIDEO) {
			push_raw_video_frame(0, frame);
		}
	}
	return true;
}
bool DummyModule::set_param(const core::Parameter& param)
{
	if (param.name == "name") {
		dummy_name = param.get<std::string>();
	} else return core::IOThread::set_param(param);
	return true;
}

} /* namespace dummy_module */
} /* namespace yuri */
