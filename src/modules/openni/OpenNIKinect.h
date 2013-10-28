/*
 * OpenNIKinect.h
 *
 *  Created on: 19.2.2013
 *      Author: neneko
 */

#ifndef OpenNIKinect_H_
#define OpenNIKinect_H_
#include "yuri/core/IOThread.h"
#include "openni_wrapper.h"

namespace yuri {
namespace OpenNIKinect {

class OpenNIKinect: public core::IOThread
{
public:
	typedef shared_ptr<openni::Device> pDevice;
	typedef shared_ptr<openni::VideoStream> pVideoStream;
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters 	configure();
	virtual 					~OpenNIKinect();
private:
	OpenNIKinect(log::Log &log_,core::pwThreadBase parent,core::Parameters &parameters);
	virtual void 				run();
	virtual bool 				set_param(const core::Parameter& param);
	bool 						enable_sensor(pDevice, openni::SensorType);


	bool 						enable_depth;
	bool 						enable_ir;
	bool 						enable_color;
	bool 						enable_sync;
	bool 						enable_registration;
	yuri::size_t 				max_sensors;
	size_t 						skip_sensors;
	std::vector<pDevice> 		devices;
	std::vector<pVideoStream> 	video_streams;
	std::vector<ssize_t> 		last_number;
};

} /* namespace OpenNIKinect */
} /* namespace yuri */
#endif /* OpenNIKinect_H_ */
