/*
 * FpsFixer.h
 *
 *  Created on: Aug 16, 2010
 *      Author: neneko
 */

#ifndef FPSFIXER_H_
#define FPSFIXER_H_

#include "yuri/core/thread/IOThread.h"
#include "yuri/event/BasicEventConsumer.h"
#include "yuri/core/utils/Timer.h"

namespace yuri {

namespace fps {

class FpsFixer: public yuri::core::IOThread, event::BasicEventConsumer
{
public:
	FpsFixer(log::Log &log_, core::pwThreadBase parent, const core::Parameters& parameters);
	virtual ~FpsFixer() noexcept;
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
private:
	virtual bool set_param(const core::Parameter &parameter) override;
	virtual void run() override;
	virtual bool do_process_event(const std::string& event_name, const event::pBasicEvent& event) override;
	double fps_;
	yuri::size_t frames_;
	Timer timer_;
};

}

}

#endif /* FPSFIXER_H_ */
