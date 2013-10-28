/*
 * FpsFixer.h
 *
 *  Created on: Aug 16, 2010
 *      Author: neneko
 */

#ifndef FPSFIXER_H_
#define FPSFIXER_H_

#include "yuri/core/IOThread.h"

//#include <boost/date_time/posix_time/posix_time.hpp>

namespace yuri {

namespace fps {
//using namespace boost::posix_time;

class FpsFixer: public yuri::core::IOThread {
public:
	FpsFixer(log::Log &log_, core::pwThreadBase parent, core::Parameters& parameters);
	virtual ~FpsFixer();
	IO_THREAD_GENERATOR_DECLARATION
	//static shared_ptr<IOThread> generate(Log &_log,pThreadBase parent) throw (Exception);
	static core::pParameters configure();
	virtual bool set_param(const core::Parameter &parameter);
	void run();
protected:
	yuri::size_t fps, fps_nom;
	time_value start_time, act_time;
	yuri::size_t frames;
};

}

}

#endif /* FPSFIXER_H_ */
