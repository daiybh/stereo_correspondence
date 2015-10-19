/*!
 * @file 		TimestampObserver.h
 * @author 		Anastasia Kuznetsova <kuzneana@gmail.com>
 * @date 		4. 5. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef PLAYBACKCONTROLLER_H_
#define PLAYBACKCONTROLLER_H_

#include "yuri/core/Module.h"

#include "yuri/event/BasicEventConsumer.h"
#include "yuri/event/BasicEventProducer.h"

namespace yuri {
namespace synchronization {


class TimestampObserver:
        public yuri::core::IOThread,
        public event::BasicEventConsumer,
        public event::BasicEventProducer
{
public:

    IOTHREAD_GENERATOR_DECLARATION
    TimestampObserver(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
    virtual ~TimestampObserver() noexcept;

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

    virtual void run() override;

    void set_timestamp();
    
private:
    bool observe_timestamp_;
    double fps_;
    bool initialized_;
    duration_t timestamp_;
};

}
}

#endif /* PLAYBACKCONTROLLER_H_ */
