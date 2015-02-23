/*
 * GPUJpegBase.cpp
 *
 *  Created on: Feb 8, 2012
 *      Author: neneko
 */

#include "GPUJpegBase.h"
#include "yuri/core/Module.h"

namespace yuri {
namespace gpujpeg {

core::Parameters GPUJpegBase::configure()
{
	core::Parameters p = core::IOFilter::configure();
	p["device"]["Cuda device id"]=0;
	return p;
}

GPUJpegBase::GPUJpegBase(log::Log &log_, core::pwThreadBase parent, const std::string& name)
:core::IOFilter(log_, parent, name),device(0)
{

}

GPUJpegBase::~GPUJpegBase() noexcept
{

}


bool GPUJpegBase::set_param(const core::Parameter& parameter)
{
	if (parameter.get_name() == "device") {
		device=parameter.get<uint16_t>();
	} else return IOFilter::set_param(parameter);
	return true;
}

bool GPUJpegBase::init_device()
{
	gpujpeg_devices_info gdi = gpujpeg_get_devices_info();
	log[log::info] << "Number of CUDA capable devices in this system: " << gdi.device_count;
	for (uint16_t i = 0; i < gdi.device_count; ++i) {
		log[log::info] << "\tDevice " << i << ": " << gdi.device[i].name;
	}
	if (gpujpeg_init_device(device,GPUJPEG_VERBOSE))
		throw exception::InitializationFailed("Failed to initialize device");
	return true;
}
} /* namespace io */
} /* namespace yuri */
