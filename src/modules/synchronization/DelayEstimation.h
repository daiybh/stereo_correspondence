/*!
 * @file 		DelayEstimation.h
 * @author 		Anastasia Kuznetsova <kuzneana@gmail.com>
 * @date 		4. 5. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef DELAYESTIMATION_H_
#define DELAYESTIMATION_H_

#include "yuri/core/Module.h"
#include "yuri/event/BasicEventProducer.h"
#include <random>

namespace yuri {
namespace synchronization {


class DelayEstimation: public yuri::core::IOThread,
                    public event::BasicEventConsumer,
                    public event::BasicEventProducer {
public:

    IOTHREAD_GENERATOR_DECLARATION
    DelayEstimation(log::Log &log_, core::pwThreadBase parent,const core::Parameters &parameters);
    virtual ~DelayEstimation() noexcept;

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

private:

    std::mt19937 gen_;
    std::uniform_int_distribution<uint64_t> dis_;
    bool changed_;
    uint64_t  last_id_;
    bool is_coordinator_;
    duration_t period_;
    duration_t timeout_;
    Timer roundtrip_dur_;

};


}
}

#endif /* DELAYESTIMATION_H_ */
