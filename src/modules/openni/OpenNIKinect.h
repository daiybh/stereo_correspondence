/*
 * OpenNIKinect.h
 *
 *  Created on: 19.2.2013
 *      Author: neneko
 */

#ifndef OpenNIKinect_H_
#define OpenNIKinect_H_
#include "yuri/io/BasicIOThread.h"
#include "openni_wrapper.h"

namespace yuri {
namespace OpenNIKinect {

class OpenNIKinect: public io::BasicIOThread
{
public:
	IO_THREAD_GENERATOR_DECLARATION
	static shared_ptr<config::Parameters> configure();
	virtual ~OpenNIKinect();
private:
	OpenNIKinect(log::Log &log_,io::pThreadBase parent,config::Parameters &parameters);
	virtual void run();
	virtual bool set_param(config::Parameter& param);
	std::string dummy_name;
};

} /* namespace OpenNIKinect */
} /* namespace yuri */
#endif /* OpenNIKinect_H_ */
