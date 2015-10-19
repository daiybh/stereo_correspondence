/*!
 * @file 		TwopcProtocolCoordinator.cpp
 * @author 		Anastasia Kuznetsova <kuzneana@gmail.com>
 * @date 		26. 3. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#include "TwopcProtocolCoordinator.h"
#include "yuri/event/EventHelpers.h"
#include <cmath>

namespace yuri {
namespace synchronization {

core::Parameters TwopcProtocolCoordinator::configure()
{
    core::Parameters p = core::IOThread::configure();
    p["cohorts"]["Count of cohorts"]=0;
    p["confirmation"]["Required count of confirmations from cohorts.[per cent]"]=100;
    p["strict"]["The frame will be displaying if all of cohorts have had frame."]=false;
    p["variable_cohorts"]["Allow variable count of cohorts."]=false;
    p["frame_index"]["Using default frame index."]=false;
    p["wait_for_replies"]["Maximum waiting time for replies from cohorts"]=1_s;
    p["missing_confirmation"]["Set the maximum respected count of missing confirmations from cohorts."]=80;
    return p;
}


IOTHREAD_GENERATOR(TwopcProtocolCoordinator)

TwopcProtocolCoordinator::TwopcProtocolCoordinator(log::Log &log_, core::pwThreadBase parent,const core::Parameters &parameters):
    core::IOThread(log_,parent,1,1,std::string("twopc_protocol_coordinator")),
    StateTransitionTable(log_, CoordinatorState::initial),
    event::BasicEventProducer(log),
    event::BasicEventConsumer(log),
    gen_(std::random_device()()), dis_(1,999999), id_(dis_(gen_)), frame_no_(0),curr_event_(CoordinatorEvent::start),
    cohorts_n_(0), confirmation_pct_(100),is_strict_(false), variable_cohorts_(false),default_frame_index_(true),
    wait_for_replies_(4_s), max_frame_delay(80)
{
    IOTHREAD_INIT(parameters)
    confirm_.expected_num = calculate_percentage(cohorts_n_, confirmation_pct_);
    log[log::info]<<"Expected conf "<< confirm_.expected_num;
    define_transition_table();
}

TwopcProtocolCoordinator::~TwopcProtocolCoordinator() noexcept{}

void TwopcProtocolCoordinator::define_transition_table(){
    add_transition(CoordinatorState::initial, CoordinatorEvent::start,
                   CoordinatorState::collecting, std::bind(&TwopcProtocolCoordinator::prepare_frame, this));

    add_transition(CoordinatorState::collecting, CoordinatorEvent::vote,
                   CoordinatorState::voting, std::bind(&TwopcProtocolCoordinator::process_replies, this));

    add_transition(CoordinatorState::voting, CoordinatorEvent::perform,
                   CoordinatorState::initial, std::bind(&TwopcProtocolCoordinator::send_perform_req, this));

    add_transition(CoordinatorState::voting, CoordinatorEvent::abort,
                   CoordinatorState::initial, std::bind(&TwopcProtocolCoordinator::send_abort_req, this));

    add_transition(CoordinatorState::initial, CoordinatorEvent::perform,
                   CoordinatorState::initial, std::bind(&TwopcProtocolCoordinator::reinc, this));

    add_transition(CoordinatorState::initial, CoordinatorEvent::abort,
                   CoordinatorState::initial, std::bind(&TwopcProtocolCoordinator::reinc, this));
}

void TwopcProtocolCoordinator::prepare_frame(){
    log[log::info] <<"Start with frame "<<cohorts_n_;
    do{
        frame_=pop_frame(0);
        if(frame_){
            if(default_frame_index_){
               frame_no_=frame_->get_index();
            }else{
                ++frame_no_;
            }
        }
    }while(still_running() && !frame_);
    curr_event_=CoordinatorEvent::vote;
}

void TwopcProtocolCoordinator::process_replies(){
    log[log::info]<<"Cohorts "<<cohorts_n_;
    Timer timer_repeat_replies;
    Timer timer_replies;
    emit_event("prepare", prepare_event(id_,frame_no_));
    do{
        if(timer_repeat_replies.get_duration() > 100_ms ){
            emit_event("prepare", prepare_event(id_,frame_no_));
            timer_repeat_replies.reset();
        }
        log[log::info]<<"wait "<<timer_repeat_replies.get_duration();
        wait_for_events(4_ms);
        process_events();
    }while(still_running() && (!is_able_do_decisison() || timer_replies.get_duration() > wait_for_replies_));

    log[log::info]<<"Confirmation expected "<<confirm_.expected_num <<" "<<confirm_.positive_num<<" "<<confirm_.negative_num;
    if(do_decision())curr_event_=CoordinatorEvent::perform;
    else curr_event_=CoordinatorEvent::abort;
    log[log::info] <<"Process replies "<<static_cast<int>(curr_event_);
}

void TwopcProtocolCoordinator::send_abort_req(){
    log[log::info]<<"Abort ";
    emit_event("abort", prepare_event(id_,frame_no_));
}

void TwopcProtocolCoordinator::send_perform_req(){
    emit_event("perform", prepare_event(id_,frame_no_));
    push_frame(0,frame_);
}


void TwopcProtocolCoordinator::run()
{
    while(still_running()){
        process_event(curr_event_);
    }
}


void TwopcProtocolCoordinator::reinc(){
    if(variable_cohorts_) {
        control_cohorts_state(cohorts_, frame_no_, max_frame_delay);
        cohorts_n_=cohorts_.size();
        confirm_.expected_num = calculate_percentage(cohorts_n_, confirmation_pct_);
    }
    curr_event_ = CoordinatorEvent::start;
    confirm_.reset();
}


bool TwopcProtocolCoordinator::is_able_do_decisison(){
    if(confirm_.positive_num >= confirm_.expected_num || confirm_.negative_num > (cohorts_n_- confirm_.expected_num)) return true;
    return false;
}


bool TwopcProtocolCoordinator::do_decision(){
    if(is_able_do_decisison()){
        if(is_strict_ && confirm_.positive_num < confirm_.expected_num) return false;
        return true;
    } else return false;
}



bool TwopcProtocolCoordinator::do_process_event(const std::string& event_name, const event::pBasicEvent& event)
{
   log[log::info]<<"Event "<<event_name;
   auto response = event::get_value<event::EventVector>(event);
   if(!verify_cohort_response(response, id_)) return false;

   uint64_t id_receiver = event::lex_cast_value<uint64_t>(response[0]);
   uint64_t id_sender = event::lex_cast_value<uint64_t>(response[1]);
   index_t frame_number = event::lex_cast_value<index_t>(response[2]);


   log[log::info]<<"Event from "<<id_sender<<" to "<<id_receiver<<" Current count of cohorts "<<cohorts_.size();

   //prijem noveho cohorta
   auto cohort = cohorts_.find(id_sender);
   if(variable_cohorts_ && (id_receiver==0 || (cohort==cohorts_.end() && id_receiver==id_))){

       cohorts_[id_sender]=frame_number;
       cohorts_n_=cohorts_.size();
       confirm_.expected_num = calculate_percentage(cohorts_.size(), confirmation_pct_);
       log[log::info]<<"Pridat cohorta "<<cohorts_n_;
   }

   if(frame_number == frame_no_ && id_receiver==id_){

       log[log::info]<<"Eventxx "<<confirm_.expected_num <<" "<<confirm_.positive_num<<" "<<confirm_.negative_num;
       if(iequals(event_name, "yes_reply")){
           ++confirm_.positive_num;
           cohorts_[id_sender]=frame_number;
       }else if(iequals(event_name, "no_reply")){
            ++confirm_.negative_num;
           cohorts_[id_sender]=frame_number;
       }
       log[log::info]<<"Eventxx "<<confirm_.expected_num <<" "<<confirm_.positive_num<<" "<<confirm_.negative_num;
   }
   if(iequals(event_name, "yes_reply")) log[log::info]<<"Global frame "<<frame_no_<<", frame number  "<<frame_number;
   return false;
}




bool TwopcProtocolCoordinator::set_param(const core::Parameter &parameter)
{
    if(assign_parameters(parameter)
            (cohorts_n_,        "cohorts")
            (confirmation_pct_, "confirmation")
            (is_strict_, "strict")
            (variable_cohorts_,    "variable_cohorts")
            (default_frame_index_, "frame_index")
            //(wait_for_replies_ , "wait_for_replies")
            (max_frame_delay ,"missing_confirmation"))
        return true;
    return core::IOThread::set_param(parameter);
}


}
}
