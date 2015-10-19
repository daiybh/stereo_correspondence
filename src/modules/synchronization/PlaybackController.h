/*!
 * @file 		PlaybackController.h
 * @author 		Anastasia Kuznetsova <kuzneana@gmail.com>
 * @date 		18. 3. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef PLAYBACKCONTROLLER_H_
#define PLAYBACKCONTROLLER_H_

#include "yuri/core/Module.h"

#include "yuri/event/BasicEventConsumer.h"
#include "yuri/event/BasicEventProducer.h"
#define TOLERANCE 0.000001

namespace yuri {
namespace synchronization {


class PlaybackController:
        public yuri::core::IOThread,
        public event::BasicEventConsumer,
        public event::BasicEventProducer
{
public:

    IOTHREAD_GENERATOR_DECLARATION
    PlaybackController(log::Log &log_, core::pwThreadBase parent,const core::Parameters &parameters);
    virtual ~PlaybackController() noexcept;

    /**
     * Set up default parameters
     */
    static core::Parameters configure();

    /**
     * Set parameters from the command-line and XML file
     * @param parameter default parameters
     * @return true if the parameter is set
     */
    virtual bool set_param(const core::Parameter &parameter) override;

    virtual bool do_process_event(const std::string& event_name, const event::pBasicEvent& event) override;

    virtual  void run() override;

    double calculate_fps(const core::pFrame frame);
    
private:

    bool is_coordinator_;
    bool paused_;
    bool stopped_;
    double fps_;
    int moved_;
    bool initialize_;

    yuri::core::pFrame frame_;
};

}
}

#endif /* PLAYBACKCONTROLLER_H_ */
