/*
 * DummyModule.cpp
 *
 *  Created on: 11.2.2013
 *      Author: neneko
 */

#include "DummyModule.h"
#include "yuri/config/RegisteredClass.h"
namespace yuri {
namespace dummy_module {

REGISTER("dummy",DummyModule)

IO_THREAD_GENERATOR(DummyModule)

using namespace yuri::io;

shared_ptr<Parameters> DummyModule::configure()
{
	shared_ptr<Parameters> p = BasicIOThread::configure();
	p->set_description("Dummy module. For testing only.");
	(*p)["size"]["Set size of ....  (ignored ;)"]=666;
	(*p)["name"]["Set name"]=std::string("");
	p->set_max_pipes(1,1);
	return p;
}


DummyModule::DummyModule(Log &log_,pThreadBase parent,Parameters &parameters):
BasicIOThread(log_,parent,1,1,std::string("dummy"))
{
	IO_THREAD_INIT("Dummy")
	if (!dummy_name.empty()) log[info] << "Got name " << dummy_name <<"\n";
}

DummyModule::~DummyModule()
{
}

bool DummyModule::step()
{
	shared_ptr<BasicFrame> frame = in[0]->pop_frame();
	if (frame) {
		yuri::format_t fmt = frame->get_format();
		if (BasicPipe::get_format_group(fmt)==YURI_TYPE_VIDEO) {
			push_raw_video_frame(0, frame);
		}
	}
	return true;
}
bool DummyModule::set_param(Parameter& param)
{
	if (param.name == "name") {
		dummy_name = param.get<std::string>();
	} else return BasicIOThread::set_param(param);
	return true;
}

} /* namespace dummy_module */
} /* namespace yuri */
