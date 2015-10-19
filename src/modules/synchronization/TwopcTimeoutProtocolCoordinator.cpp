/*!
 * @file 		TwopcTimeoutProtocolCoordinator.cpp
 * @author 		Anastasia Kuznetsova <kuzneana@gmail.com>
 * @date 		26. 3. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#include "TwopcTimeoutProtocolCoordinator.h"
#include "yuri/event/EventHelpers.h"

namespace yuri{
namespace synchronization{

core::Parameters TwopcTimeoutProtocolCoordinator::configure()
{
    core::Parameters p = core::IOThread::configure();
    p["fps"]["Override framerate. Specify 0 to use original, or negative value to maximal speed."]=0;
    p["strict"]["The frame will be displaying if all of cohorts have had frame."]=false;
    p["missing_confirmation"]["Set the maximum respected count of missing confirmations from cohorts."]=7;
    p["frame_index"]["Using default frame index."]=false;
    //p["timeout"]["Maximum waiting time for replies from cohorts"]=19_ms;
    return p;
}


IOTHREAD_GENERATOR(TwopcTimeoutProtocolCoordinator)

TwopcTimeoutProtocolCoordinator::TwopcTimeoutProtocolCoordinator(log::Log &log_, core::pwThreadBase parent,const core::Parameters &parameters):
    core::IOThread(log_,parent,1,1,std::string("twopc_timeout_coordinator")),
    StateTransitionTable(log_,TimeoutCoordinatorState::initial),
    event::BasicEventProducer(log),
    event::BasicEventConsumer(log),
    gen_(std::random_device()()), distrib_(1,999999), id_(distrib_(gen_)),
    frame_no_(0), fps_(0),is_strict_(false),frame_delay_(20),default_frame_index_(true), timeout_(800_ms), curr_event_(TimeoutCoordinatorEvent::start),
    initialize_(false)
{
    IOTHREAD_INIT(parameters)
    define_transition_table();
}

TwopcTimeoutProtocolCoordinator::~TwopcTimeoutProtocolCoordinator() noexcept{}


void TwopcTimeoutProtocolCoordinator::define_transition_table(){
add_transition(TimeoutCoordinatorState::initial, TimeoutCoordinatorEvent::start,
              TimeoutCoordinatorState::collecting, std::bind(&TwopcTimeoutProtocolCoordinator::prepare_frame, this));

add_transition(TimeoutCoordinatorState::collecting, TimeoutCoordinatorEvent::vote,
              TimeoutCoordinatorState::voiting, std::bind(&TwopcTimeoutProtocolCoordinator::process_replies, this));

add_transition(TimeoutCoordinatorState::voiting, TimeoutCoordinatorEvent::abort,
              TimeoutCoordinatorState::initial, std::bind(&TwopcTimeoutProtocolCoordinator::do_abort, this));

add_transition(TimeoutCoordinatorState::voiting, TimeoutCoordinatorEvent::perform,
              TimeoutCoordinatorState::initial, std::bind(&TwopcTimeoutProtocolCoordinator::do_perform, this));

add_transition(TimeoutCoordinatorState::initial, TimeoutCoordinatorEvent::abort,
              TimeoutCoordinatorState::initial, std::bind(&TwopcTimeoutProtocolCoordinator::reinc, this));
add_transition(TimeoutCoordinatorState::initial, TimeoutCoordinatorEvent::perform,
              TimeoutCoordinatorState::initial, std::bind(&TwopcTimeoutProtocolCoordinator::reinc, this));
}



void TwopcTimeoutProtocolCoordinator::prepare_frame(){
    //log[log::info] <<"Start with frame ";
    do {
        frame_=pop_frame(0);
        if(frame_){
            if(default_frame_index_){
               frame_no_=frame_->get_index();
            }else{
                ++frame_no_;
            }
        }
    } while (still_running() && !frame_);
    if (!initialize_){
        if(fps_ < TOLERANCE) fps_ = calculate_fps(frame_);
        timeout_ = set_timeout(fps_, timeout_, log, 1);
        initialize_ = true;
    }
    curr_event_ = TimeoutCoordinatorEvent::vote;
    synch_timeout_.reset();
    emit_event("prepare", prepare_event(id_, frame_no_));
}

void TwopcTimeoutProtocolCoordinator::process_replies(){

    do{
        log[log::info]<<"Timeout "<<timeout_;
        wait_for_events(4_ms);
        process_events();
   }while((confirm_.positive_num!=cohorts_.size() || cohorts_.size()==0) && synch_timeout_.get_duration() < timeout_
          && still_running() && confirm_.negative_num!=cohorts_.size() );

    log[log::info] <<"Fps "<<fps_;
    log[log::info] <<"Process replies "<<cohorts_.size()<<" succcs " << confirm_.positive_num <<" negative " << confirm_.negative_num;

    log[log::info] <<"Perform frame "<<frame_no_<<" timer: "<<synch_timeout_.get_duration()<<" timeout: "<<timeout_;
    if(do_decision())curr_event_=TimeoutCoordinatorEvent::perform;
    else curr_event_=TimeoutCoordinatorEvent::abort;
    log[log::info]<<"wait for replies "<<synch_timeout_.get_duration()<<" - "<<timeout_;
}

void TwopcTimeoutProtocolCoordinator::do_abort(){
    //log[log::info]<<"Abort ";
    emit_event("abort", prepare_event(id_, frame_no_));
}

void TwopcTimeoutProtocolCoordinator::do_perform(){
    log[log::info] <<"Perform frame "<<frame_no_<<" timer: "<<synch_timeout_.get_duration()<<" timeout: "<<timeout_;
    emit_event("perform", prepare_event(id_, frame_no_));
    push_frame(0,frame_);
}

void TwopcTimeoutProtocolCoordinator::reinc(){
    control_cohorts_state(cohorts_, frame_no_, frame_delay_);
    confirm_.reset();
    curr_event_=TimeoutCoordinatorEvent::start;
}

void TwopcTimeoutProtocolCoordinator::run()
{
    while(still_running()){
        //log[log::info]<<"Event "<<&curr_event_;
        process_event(curr_event_);
    }
}

bool TwopcTimeoutProtocolCoordinator::do_decision(){
    if(is_strict_ && (confirm_.positive_num!=cohorts_.size() || confirm_.negative_num>0)) return false;
    return true;
}


bool TwopcTimeoutProtocolCoordinator::do_process_event(const std::string& event_name, const event::pBasicEvent& event)
{
     //log[log::info]<<"Event "<<event_name;

    auto response = event::get_value<event::EventVector>(event);
    if(!verify_cohort_response(response, id_)) return false;

    uint64_t id_receiver = event::lex_cast_value<uint64_t>(response[0]);
    uint64_t id_sender = event::lex_cast_value<uint64_t>(response[1]);
    index_t frame_number = event::lex_cast_value<index_t>(response[2]);


    //log[log::info]<<"Event from "<<id_sender<<" to "<<id_receiver<<" Current count of cohorts "<<cohorts_.size();

    //prijem noveho cohorta
    if(id_receiver==0 || cohorts_.find(id_sender)==cohorts_.end()){
        cohorts_[id_sender]=frame_number;
    }

    if(frame_number == frame_no_){
        if(iequals(event_name, "yes_reply")){
           ++confirm_.positive_num;
        }else if(iequals(event_name, "no_reply")){
            ++confirm_.negative_num;
        }
        cohorts_[id_sender]=frame_number;
    }
    return false;
}

bool TwopcTimeoutProtocolCoordinator::set_param(const core::Parameter &parameter)
{

    if(assign_parameters(parameter)
            (fps_, "fps")
            (is_strict_, "strict")
            (frame_delay_ ,"missing_confirmation")
            (default_frame_index_, "frame_index")
            //(timeout_, "timeout")
            )
        return true;
    return core::IOThread::set_param(parameter);
}

}
}
