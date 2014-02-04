/*
 * FpsFixer.cpp
 *
 *  Created on: Aug 16, 2010
 *      Author: neneko
 */

#include "FpsFixer.h"
#include "yuri/core/Module.h"
#include "yuri/core/utils/Timer.h"

namespace yuri {

namespace fps {


MODULE_REGISTRATION_BEGIN("fix_fps")
	REGISTER_IOTHREAD("fix_fps",FpsFixer)
MODULE_REGISTRATION_END()

IOTHREAD_GENERATOR(FpsFixer)

core::Parameters FpsFixer::configure()
{
	core::Parameters p = core::IOThread::configure();
	p["fps"]["FPS"]=25.0;
//	p["fps_nom"]["FPS nominator. ie. frame wil be output once every fps_nom/fps seconds"]=1;
	return p;
}


FpsFixer::FpsFixer(log::Log &log_, core::pwThreadBase parent, const core::Parameters& parameters):
		IOThread(log_,parent,1,1,"FpsFixer"),fps_(25.0)
{
	IOTHREAD_INIT(parameters)
}

FpsFixer::~FpsFixer() noexcept
{
//	log[log::info] << "Outputed " << frames << " in " << to_simple_string(act_time - start_time) <<
//				". That makes " << (double)frames*1.0e6/((act_time - start_time).total_microseconds()) <<
//				"frames/s\n";
}

void FpsFixer::run()
{
	IOThread::print_id();
	core::pFrame frame;
	Timer timer;

	frames = 0;
	while(still_running()) {
		while (auto f = pop_frame(0)) {
			frame = f;
		}
		if (timer.get_duration() > frames * 1_s/fps_) {
			if (frame) {
				push_frame(0,frame);
			}
			frames++;
		} else {
			sleep(get_latency());
		}
	}

}

bool FpsFixer::set_param(const core::Parameter &parameter)
{
	if (parameter.get_name() == "fps") {
		fps_=parameter.get<double>();
	} else return IOThread::set_param(parameter);
	return true;
}
}

}
