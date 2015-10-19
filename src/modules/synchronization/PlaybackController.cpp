/*!
 * @file 		PlaybackController.cpp
 * @author 		Anastasia Kuznetsova <kuzneana@gmail.com>
 * @date 		18. 3. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#include "PlaybackController.h"
#include "yuri/core/utils/assign_events.h"

namespace yuri {
namespace synchronization {

core::Parameters PlaybackController::configure()
{
    core::Parameters p = core::IOThread::configure();
    p["coordinator"]["Sets coordinator mode"]=false;
    p["fps"]["Override framerate. Specify 0 to use original, or negative value to maximal speed."]=0;
    return p;
}


IOTHREAD_GENERATOR(PlaybackController)


PlaybackController::PlaybackController(log::Log &log_, core::pwThreadBase parent,const core::Parameters &parameters):
    core::IOThread(log_,parent,1,1,std::string("playback_controller")),event::BasicEventConsumer(log), event::BasicEventProducer(log),
    is_coordinator_(false), paused_(true), stopped_(true), fps_(0.0), moved_(0), initialize_(false)
{
    IOTHREAD_INIT(parameters)
}

PlaybackController::~PlaybackController() noexcept{}


void PlaybackController::run()
{
    Timer t2;
    while(still_running()){
        wait_for_events(12_ms);
        process_events();
        //log[log::info]<<"Paused "<<paused_;
        if(paused_)continue;
        if(moved_!=0 && fps_ > TOLERANCE){
            emit_event("observe_timestamp", false);
            //log[log::info]<<"Moved= "<<moved_;
            int i=0;
            Timer t;
            while(still_running() && i<moved_){
                frame_ = pop_frame(0);
                if(frame_){
                    i++;
//                    log[log::info]<<"Moving "<<frame_<<" "<<frame_.get()<<" dur: "<<t.get_duration();
                    push_frame(0, std::move(frame_));

                    t.reset();
                }

            }

            emit_event("observe_timestamp", true);
            moved_=0;
        }else{
            frame_ = pop_frame(0);
            if (!frame_) continue;
            t2.reset();
        }
        if(!initialize_){
            if(fps_<TOLERANCE) fps_ = calculate_fps(frame_);
            initialize_=true;
        }
        push_frame(0, frame_);
    }

}

double PlaybackController::calculate_fps(const core::pFrame frame){
    if(frame->get_duration().value==0) return 0.0;
    return (1_s).value/frame->get_duration().value;
}

bool PlaybackController::do_process_event(const std::string& event_name, const event::pBasicEvent& event)
{
    if(event_name == "move"){
        int move_min = event::lex_cast_value<int>(event);
        moved_ = move_min*60*fps_;
        return true;
    }
    if(assign_events(event_name, event)
            (is_coordinator_, "coordinator")
            (fps_, "fps")
            (stopped_, "stop")
            (paused_, "pause")
            )
        return true;
    return false;
}


bool PlaybackController::set_param(const core::Parameter &parameter)
{
    if(assign_parameters(parameter)
            (is_coordinator_, "coordinator")
            (fps_, "fps"))
        return true;
    return IOThread::set_param(parameter);
}

}
}
