/*
 * OpenNIKinect.cpp
 *
 *  Created on: 19.2.2013
 *      Author: neneko
 */

#include "OpenNIKinect.h"
#include "yuri/config/RegisteredClass.h"
namespace yuri {
namespace OpenNIKinect {

REGISTER("openni_kinect",OpenNIKinect)

IO_THREAD_GENERATOR(OpenNIKinect)


shared_ptr<config::Parameters> OpenNIKinect::configure()
{
	shared_ptr<config::Parameters> p = BasicIOThread::configure();
	p->set_description("OpenNIKinect Kinect source.");
	/*(*p)["size"]["Set size of ....  (ignored ;)"]=666;
	(*p)["name"]["Set name"]=std::string("");*/
	p->set_max_pipes(0,1);
	return p;
}

OpenNIKinect::OpenNIKinect(log::Log &log_,io::pThreadBase parent, config::Parameters &parameters):
io::BasicIOThread(log_,parent,1,1,std::string("OpenNIKinect"))
{
	IO_THREAD_INIT("OpenNIKinect")
	//if (!dummy_name.empty()) log[info] << "Got name " << dummy_name <<"\n";
	if (openni::OpenNI::initialize()!=  openni::STATUS_OK) {
		log[log::fatal]<<"Failed to initialize OpenNI! Error: "<< openni::OpenNI::getExtendedError() <<"\n";
		throw yuri::exception::InitializationFailed("Failed to initialize OpenNI!");
	}
	openni::Array< openni::DeviceInfo > devices;
	openni::OpenNI::enumerateDevices(&devices);
	if (!devices.getSize()) {
		log[log::fatal]<<"No devices found!\n";
		throw yuri::exception::InitializationFailed("No devices found!");
	}
	for (int i=0;i<devices.getSize();++i) {
		log[log::info] << "Device " << i <<": " << devices[i].getName() << ", uri: " << devices[i].getUri() << "\n";
	}

	openni::OpenNI::shutdown();
	throw yuri::exception::InitializationFailed("Not implemented!");
}
OpenNIKinect::~OpenNIKinect()
{
}

void OpenNIKinect::run()
{
	IO_THREAD_PRE_RUN

	IO_THREAD_POST_RUN
}
bool OpenNIKinect::set_param(config::Parameter& param)
{
	/*if (param.name == "name") {
		dummy_name = param.get<std::string>();
	} else */return BasicIOThread::set_param(param);
//	return true;
}


} /* namespace OpenNIKinect */
} /* namespace yuri */
