/*!
 * @file 		TwopcProtocolCohort.cpp
 * @author 		Anastasia Kuznetsova <kuzneana@gmail.com>
 * @date 		26. 3. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#include "TwopcProtocolCohort.h"

namespace yuri {
namespace synchronization {

core::Parameters TwopcProtocolCohort::configure()
{
    core::Parameters p = core::IOThread::configure();
    p["waiting_frame"]["It is the maximal period in which frame must be prepared. Change this period in the case of untrustworthy transfer."]="milliseconds(1)";
    p["frame_index"]["Using default frame index."]=false;
    return p;
}

IOTHREAD_GENERATOR(TwopcProtocolCohort)


TwopcProtocolCohort::TwopcProtocolCohort(log::Log &log_, core::pwThreadBase parent,const core::Parameters &parameters):
    core::IOThread(log_,parent,1,1,std::string("twopc_protocol_cohort")),
    StateTransitionTable(log_, CohortState::initial),
    event::BasicEventProducer(log),
    event::BasicEventConsumer(log),
    gen_(std::random_device()()), dis_(1,999999), id_(dis_(gen_)), id_coordinator_(0),
    local_frame_no_(0), global_frame_no_(-1), curr_event_(CohortEvent::start), default_frame_index_(true), waiting_for_frame_(10_ms)
{
    IOTHREAD_INIT(parameters)
    define_transition_table();
}

TwopcProtocolCohort::~TwopcProtocolCohort() noexcept{}

void TwopcProtocolCohort::define_transition_table()
{
    add_transition(CohortState::initial, CohortEvent::start,
                   CohortState::collecting, std::bind(&TwopcProtocolCohort::wait_for_prepare , this));
    add_transition(CohortState::initial, CohortEvent::vote_no,
                   CohortState::collecting, std::bind(&TwopcProtocolCohort::wait_for_prepare , this));
    add_transition(CohortState::initial, CohortEvent::perform,
                   CohortState::collecting, std::bind(&TwopcProtocolCohort::wait_for_prepare , this));
    add_transition(CohortState::initial, CohortEvent::abort,
                   CohortState::collecting, std::bind(&TwopcProtocolCohort::wait_for_prepare , this));


    add_transition(CohortState::collecting, CohortEvent::prepare,
                   CohortState::voiting, std::bind(&TwopcProtocolCohort::prepare_frame , this));

    add_transition(CohortState::voiting, CohortEvent::vote_yes,
                   CohortState::prepared, std::bind(&TwopcProtocolCohort::send_vote , this));
    add_transition(CohortState::voiting, CohortEvent::vote_no,
                   CohortState::initial, std::bind(&TwopcProtocolCohort::send_vote , this));

    add_transition(CohortState::prepared, CohortEvent::vote_yes,
                   CohortState::voiting, std::bind(&TwopcProtocolCohort::wait_for_decision , this));

    add_transition(CohortState::voiting, CohortEvent::perform,
                   CohortState::initial, std::bind(&TwopcProtocolCohort::perform , this));
    add_transition(CohortState::voiting, CohortEvent::abort,
                   CohortState::initial, std::bind(&TwopcProtocolCohort::do_default_action, this));
    add_transition(CohortState::voiting, CohortEvent::prepare,
                   CohortState::collecting, std::bind(&TwopcProtocolCohort::do_default_action, this));
}


void TwopcProtocolCohort::wait_for_prepare()
{
    //log[log::info] <<"Wait for prepare ";
    //CohortEvent last_event = curr_event_;
    while(curr_event_!=CohortEvent::prepare && still_running()){
        wait_for_events(3_ms);
        process_events();
    }
}


void TwopcProtocolCohort::prepare_frame()
{
    //log[log::info] <<"Prepare frame";
    //Timer frame_timeout;
    do{
        frame_=pop_frame(0);
        if(frame_){
            if(default_frame_index_){
               local_frame_no_=frame_->get_index();
            }else{
                ++local_frame_no_;
            }
        }
    }while(still_running() && local_frame_no_!=global_frame_no_ );
    if(frame_){
        curr_event_=CohortEvent::vote_yes;
    } else {
        curr_event_=CohortEvent::vote_no;
    }
}

void TwopcProtocolCohort::send_vote()
{
    if(frame_){
        emit_event("yes", prepare_event(id_, local_frame_no_, id_coordinator_, true));
    } else {
        emit_event("no", prepare_event(id_, local_frame_no_, id_coordinator_, true));
    }
    //log[log::info] <<"Frame "<<frame_<<", local="<<local_frame_no_;
}

void TwopcProtocolCohort::wait_for_decision()
{
    log[log::info] <<"Wait for perform";
    CohortEvent last_event = curr_event_;
    while(last_event==curr_event_ && still_running()){
        wait_for_events(20_ms);
        process_events(0);
    }
    ////log[log::info] <<"Last event "<<static_cast<int>(last_event) <<" to "<<static_cast<int>(curr_event_);
}


void TwopcProtocolCohort::perform()
{
    ////log[log::info] <<"Perform";
    push_frame(0, frame_);
    curr_event_=CohortEvent::start;
}

void TwopcProtocolCohort::run()
{
    while(still_running()){
        process_event(curr_event_);
    }
}

bool TwopcProtocolCohort::do_process_event(const std::string& event_name, const event::pBasicEvent& event)
{
    auto response = event::get_value<event::EventVector>(event);
    if(!verify_coordinator_response(response, id_coordinator_)) return false;
log[log::info] <<"eVENT "<<event_name;
    if(id_coordinator_ == 0) id_coordinator_=event::lex_cast_value<uint64_t>(response[0]);
    index_t frame_number = event::lex_cast_value<index_t>(response[1]);


    if(iequals(event_name, "prepare_req") && global_frame_no_!=frame_number){
        global_frame_no_ = frame_number;
        curr_event_=CohortEvent::prepare;
    }else if(iequals(event_name, "perform_req") && frame_number==local_frame_no_){
        curr_event_=CohortEvent::perform;
    }else if(iequals(event_name, "abort_req") && frame_number==local_frame_no_){
        curr_event_=CohortEvent::abort;
    }
    return false;
}



bool TwopcProtocolCohort::set_param(const core::Parameter &parameter)
{
    if(assign_parameters(parameter)
            (waiting_for_frame_, "waiting_frame")
            (default_frame_index_, "frame_index"))
        return true;
    return core::IOThread::set_param(parameter);
}

}
}
