/*!
 * @file 		DelayEstimation.cpp
 * @author 		Anastasia Kuznetsova <kuzneana@gmail.com>
 * @date 		4. 5. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#include "DelayEstimation.h"
#include "yuri/event/EventHelpers.h"

namespace yuri {
namespace synchronization {

core::Parameters DelayEstimation::configure()
{
    core::Parameters p = core::IOThread::configure();
    p["coordinator"]["Sets coordinator mode"] = false;
    p["period"]["Sets the duration of one cycle. [ms]"]=2000;
    p["timeout"]["Sets maximal waiting period for a response from the coordinator. [ms]"]=2000;
    return p;
}


IOTHREAD_GENERATOR(DelayEstimation)

DelayEstimation::DelayEstimation(log::Log &log_, core::pwThreadBase parent,const core::Parameters &parameters):
    core::IOThread(log_,parent,1,1,std::string("delay_estimation")), event::BasicEventConsumer(log), event::BasicEventProducer(log),
    gen_(std::random_device()()), dis_(1,999999), changed_(false), last_id_(0),
    is_coordinator_(false),  period_(2_s), timeout_(2_s)
{
    IOTHREAD_INIT(parameters)
}

DelayEstimation::~DelayEstimation() noexcept{}


void DelayEstimation::run()
{
    while(still_running()){
        if(is_coordinator_){
            wait_for_events(5_s);
            process_events();
        }else{
            changed_ = false;
            last_id_ = dis_(gen_);
            roundtrip_dur_.reset();
            emit_event("connection_test", last_id_);
            Timer waiting_period;
            while(!changed_){
                wait_for_events(500_ms);
                process_events();
                if (waiting_period.get_duration() > timeout_ || !still_running()) break;
            }
            sleep(period_);
        }

    }
}



bool DelayEstimation::do_process_event(const std::string& event_name, const event::pBasicEvent& event)
{
    if(is_coordinator_){
       if(event_name == "connection_test"){
           emit_event("connection_reply", event);
           return true;
       }
    }else{
       if (event_name == "connection_reply"){
           auto new_id = event::lex_cast_value<uint64_t>(event);
           log[log::debug] << "Receive reply " << new_id << ", expected " << last_id_;
           if (new_id == last_id_){
               duration_t dur_vl = roundtrip_dur_.get_duration()/2;
               emit_event("delay", dur_vl.value/1000);
               last_id_=0;
               changed_ = true;
           }
           return true;
       }
    }
    return false;
}


bool DelayEstimation::set_param(const core::Parameter &parameter)
{
    if(assign_parameters(parameter)
            (is_coordinator_, "coordinator")
            .parsed<float>
    			(period_, "period", [](float f){return 1_ms*f;})
			.parsed<float>
            (timeout_, "timeout", [](float f){return 1_ms*f;})
            )
        return true;
    return core::IOThread::set_param(parameter);
}

}
}
