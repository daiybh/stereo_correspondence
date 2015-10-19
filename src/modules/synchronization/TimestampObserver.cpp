/*!
 * @file 		TimestampObserver.cpp
 * @author 		Anastasia Kuznetsova <kuzneana@gmail.com>
 * @date 		4. 5. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#include "TimestampObserver.h"
#include "yuri/core/utils/assign_events.h"

namespace yuri{
namespace synchronization {

constexpr double fps_tolerance = 0.000001;

core::Parameters TimestampObserver::configure()
{
    core::Parameters p = core::IOThread::configure();
    p["observe_timestamp"]["Observe timestamps"]=true;
    p["fps"]["Specify framerate"]=25;
    return p;
}


IOTHREAD_GENERATOR(TimestampObserver)


TimestampObserver::TimestampObserver(log::Log &log_, core::pwThreadBase parent,const core::Parameters &parameters):
    core::IOThread(log_,parent,1,1,std::string("timestamp_observer")),event::BasicEventConsumer(log), event::BasicEventProducer(log),
    observe_timestamp_(true), fps_(0.0), initialized_(false)
{
    IOTHREAD_INIT(parameters)
}

TimestampObserver::~TimestampObserver() noexcept{}


void TimestampObserver::run()
{
    Timer delta;
    while(still_running()){
        process_events();
        auto frame_ = pop_frame(0);
        if (!frame_) continue;
        if(!initialized_){
            set_timestamp();
            initialized_=true;
        }
        while(still_running() && observe_timestamp_ && delta.get_duration() < timestamp_) {
        	process_events();
        	sleep(0.1_ms);
        }
        push_frame(0, frame_);
        delta.reset();
    }

}

void TimestampObserver::set_timestamp(){
    if(fps_ > fps_tolerance) {
        timestamp_= 1_s / fps_;
    }else timestamp_= 1_s / 25;
}

bool TimestampObserver::do_process_event(const std::string& event_name, const event::pBasicEvent& event)
{
    if(assign_events(event_name, event)
            (fps_, "fps")
            (observe_timestamp_, "observe_timestamp")
            ) {
    	initialized_ = false;
        return true;
    }
    return false;
}


bool TimestampObserver::set_param(const core::Parameter &parameter)
{
    if(assign_parameters(parameter)
            (observe_timestamp_, "observe_timestamp")
            (fps_, "fps"))
        return true;
    return IOThread::set_param(parameter);
}

}
}
