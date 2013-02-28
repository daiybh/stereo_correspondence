/*
 * FpsFixer.cpp
 *
 *  Created on: Aug 16, 2010
 *      Author: neneko
 */

#include "FpsFixer.h"
#include "yuri/core/Module.h"
namespace yuri {

namespace fps {


REGISTER("fix_fps",FpsFixer)

IO_THREAD_GENERATOR(FpsFixer)

core::pParameters FpsFixer::configure()
{
	core::pParameters p = core::BasicIOThread::configure();
	p->set_max_pipes(1,1);
	(*p)["fps"]["FPS"]=25;
	(*p)["fps_nom"]["FPS nominator. ie. frame wil be output once every fps_nom/fps seconds"]=1;
	return p;
}


FpsFixer::FpsFixer(log::Log &log_, core::pwThreadBase parent, core::Parameters& parameters):
		BasicIOThread(log_,parent,1,1,"FpsFixer"),fps(fps)
{
	IO_THREAD_INIT("FpsFixer")
}

FpsFixer::~FpsFixer()
{
	log[log::info] << "Outputed " << frames << " in " << to_simple_string(act_time - start_time) <<
				". That makes " << (double)frames*1.0e6/((act_time - start_time).total_microseconds()) <<
				"frames/s\n";
}

void FpsFixer::run()
{
	BasicIOThread::print_id();
	core::pBasicFrame frame;
	time_duration time_delta = microseconds(fps_nom*1e6/fps);
	yuri::size_t act_index = 0;
	ptime next_time = microsec_clock::local_time();
	start_time = next_time;
	time_duration offset = next_time.time_of_day();
	offset = microseconds(offset.total_microseconds() % time_delta.total_microseconds());
	next_time = next_time - offset + time_delta;
	frames = 0;
	while(still_running()) {
		if (in[0] && !in[0]->is_empty()) {
			while (!in[0]->is_empty()) frame=in[0]->pop_frame();
		}
		act_time=microsec_clock::local_time();
		if (act_time > next_time) {
			if (frame) {
				if (out[0]) push_raw_video_frame(0,frame);//->get_copy());
				log[log::debug] << "Pushed frame in " << to_simple_string(act_time.time_of_day()) <<
						", time_delta is " << to_simple_string(time_delta) << ", next_time was: " <<
						to_simple_string(next_time.time_of_day()) << "\n";

			}
			next_time+=time_delta;
			frames++;
		} else {
			ThreadBase::sleep((next_time - act_time).total_microseconds()/2);
		}
	}

}

bool FpsFixer::set_param(const core::Parameter &parameter)
{
	if (parameter.name == "fps") {
		fps=parameter.get<yuri::size_t>();
	} else if (parameter.name == "fps_nom") {
		fps_nom=parameter.get<yuri::size_t>();
	} else return BasicIOThread::set_param(parameter);
	return true;
}
}

}
