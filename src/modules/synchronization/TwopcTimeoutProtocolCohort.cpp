/*!
 * @file 		TwopcTimeoutProtocolCohort.cpp
 * @author 		Anastasia Kuznetsova <kuzneana@gmail.com>
 * @date 		26. 3. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#include "TwopcTimeoutProtocolCohort.h"
#include "yuri/event/EventHelpers.h"
#include "yuri/core/forward.h"

namespace yuri {
namespace synchronization {

core::Parameters TwopcTimeoutProtocolCohort::configure()
{
    core::Parameters p = core::IOThread::configure();
    p["fps"]["Override framerate. Specify 0 to use original, or negative value to maximal speed."]=0;
    p["frame_index"]["Using default frame index."]=false;
    p["timeout"]["Maximum waiting time for replies from coordinator"]="milliseconds(19)";
    return p;
}

IOTHREAD_GENERATOR(TwopcTimeoutProtocolCohort)


TwopcTimeoutProtocolCohort::TwopcTimeoutProtocolCohort(log::Log &log_, core::pwThreadBase parent,const core::Parameters &parameters):
    core::IOThread(log_,parent,1,1,std::string("twopc_timeout_cohort")),
    StateTransitionTable(log_, TimeoutCohortState::initial),
    event::BasicEventProducer(log),
    event::BasicEventConsumer(log),
    gen_(std::random_device()()), distrib_(1,999999), id_(distrib_(gen_)), id_coordinator_(0),
    global_frame_no_(1), local_frame_no_(0), fps_(0.0), curr_event_(TimeoutCohortEvent::start), initialize_(false), default_frame_index_(true),
    timeout_(800_ms)
{
    IOTHREAD_INIT(parameters)
    define_transition_table();
}

TwopcTimeoutProtocolCohort::~TwopcTimeoutProtocolCohort() noexcept{}


void TwopcTimeoutProtocolCohort::define_transition_table(){
//log[log::info] <<" Define transition " ;

add_transition(TimeoutCohortState::initial, TimeoutCohortEvent::start,
               TimeoutCohortState::collecting, std::bind(&TwopcTimeoutProtocolCohort::wait_for_prepare, this));

add_transition(TimeoutCohortState::collecting, TimeoutCohortEvent::prepare,
               TimeoutCohortState::voiting, std::bind(&TwopcTimeoutProtocolCohort::prepare_frame, this));

add_transition(TimeoutCohortState::voiting, TimeoutCohortEvent::vote_yes,
               TimeoutCohortState::prepared, std::bind(&TwopcTimeoutProtocolCohort::wait_for_decision, this));

add_transition(TimeoutCohortState::voiting, TimeoutCohortEvent::vote_no,
               TimeoutCohortState::collecting, std::bind(&TwopcTimeoutProtocolCohort::wait_for_prepare, this));

add_transition(TimeoutCohortState::prepared, TimeoutCohortEvent::perform,
               TimeoutCohortState::collecting, std::bind(&TwopcTimeoutProtocolCohort::perform, this));


add_transition(TimeoutCohortState::prepared, TimeoutCohortEvent::timeout,
               TimeoutCohortState::collecting, std::bind(&TwopcTimeoutProtocolCohort::perform, this));


add_transition(TimeoutCohortState::voiting, TimeoutCohortEvent::prepare,
               TimeoutCohortState::voiting, std::bind(&TwopcTimeoutProtocolCohort::prepare_frame, this));


add_transition(TimeoutCohortState::prepared, TimeoutCohortEvent::prepare,
               TimeoutCohortState::voiting, std::bind(&TwopcTimeoutProtocolCohort::prepare_frame, this));


add_transition(TimeoutCohortState::prepared, TimeoutCohortEvent::abort,
               TimeoutCohortState::collecting, std::bind(&TwopcTimeoutProtocolCohort::wait_for_prepare, this));


}



void TwopcTimeoutProtocolCohort::prepare_frame(){
    timer_.reset();
    //log[log::info] <<"Start with frame "<<frame_;
    do{
        frame_=pop_frame(0);
        if(frame_){
            if(default_frame_index_){
               local_frame_no_=frame_->get_index();
            }else{
                ++local_frame_no_;
            }
        }
    }while(still_running() && local_frame_no_<global_frame_no_);
    if(!initialize_){
        if(fps_<TOLERANCE) fps_ = calculate_fps(frame_);
        log[log::info] <<"Fps "<<fps_<<" "<<frame_;
        timeout_ = set_timeout(fps_, timeout_, log_, 1);
        initialize_=true;
    }


    //log[log::info] <<"Frame "<<frame_<<", global="<< global_frame_no_  <<", local="<<local_frame_no_<<" "<<timer_.get_duration();
    if(frame_){
        emit_event("yes", prepare_event(id_, local_frame_no_, id_coordinator_, true));
        curr_event_=TimeoutCohortEvent::vote_yes;
    } else {
        emit_event("no", prepare_event(id_, local_frame_no_, id_coordinator_, true));
        curr_event_=TimeoutCohortEvent::vote_no;
    }
    //log[log::info] <<"Frame "<<frame_<<", local="<<local_frame_no_;
}

void TwopcTimeoutProtocolCohort::wait_for_decision(){
    //log[log::info] <<"Wait for decision, duration "<<timer_.get_duration();
    TimeoutCohortEvent last_event = curr_event_;
    while(timer_.get_duration() < timeout_ && last_event==curr_event_ && still_running()){
        wait_for_events(3_ms);
        process_events(0);
    }
    if(timer_.get_duration() >= timeout_)curr_event_=TimeoutCohortEvent::timeout;
    log[log::info] <<"Last event "<<static_cast<int>(last_event) <<" to "<<static_cast<int>(curr_event_);
}


void TwopcTimeoutProtocolCohort::wait_for_prepare(){
    //log[log::info] <<"Wait for prepare, duration "<<timer_.get_duration();

    while(curr_event_!=TimeoutCohortEvent::prepare && still_running()){
        wait_for_events(3_ms);
        process_events();
    }
    //log[log::info] <<"Last event "<<static_cast<int>(last_event) <<" to "<<static_cast<int>(curr_event_);
}


void TwopcTimeoutProtocolCohort::perform(){
    log[log::info] <<"Perform timer: "<<timer_.get_duration()<<" timeout: "<<timeout_;
    push_frame(0, frame_);
    wait_for_prepare();
}



void TwopcTimeoutProtocolCohort::run(){
    while(still_running()){
        process_event(curr_event_);
    }
}


bool TwopcTimeoutProtocolCohort::do_process_event(const std::string& event_name, const event::pBasicEvent& event)
{
    //log[log::info]<<"EVENT "<<event_name;

    auto response = event::get_value<event::EventVector>(event);
    if(!verify_coordinator_response(response, id_coordinator_))  return false;

    if(id_coordinator_ == 0) id_coordinator_=event::lex_cast_value<uint64_t>(response[0]);
    index_t frame_number = event::lex_cast_value<index_t>(response[1]);


    if(iequals(event_name, "prepare_req")){
        global_frame_no_ = frame_number;
        curr_event_=TimeoutCohortEvent::prepare;
    }else if(frame_number == global_frame_no_){
        if(iequals(event_name, "perform_req")){
            curr_event_=TimeoutCohortEvent::perform;
        }else if(iequals(event_name, "abort_req")){
            curr_event_=TimeoutCohortEvent::abort;
        }
    } else return false;
    return true;
}

bool TwopcTimeoutProtocolCohort::set_param(const core::Parameter &parameter)
{
    if(assign_parameters(parameter)
            (fps_,     "fps")
            (default_frame_index_, "frame_index")
            //(timeout_, "timeout")
            )
        return true;
    return core::IOThread::set_param(parameter);
}


}
}
