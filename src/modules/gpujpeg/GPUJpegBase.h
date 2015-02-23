/*
 * GPUJpegBase.h
 *
 *  Created on: Feb 8, 2012
 *      Author: neneko
 */

#ifndef GPUJPEGBASE_H_
#define GPUJPEGBASE_H_
extern "C" {
	#include "libgpujpeg/gpujpeg.h"
}
#include "yuri/core/thread/IOFilter.h"
//#include "yuri/config/RegisteredClass.h"

namespace yuri {
namespace gpujpeg {

class GPUJpegBase: public core::IOFilter {
public:
	static core::Parameters configure();

	virtual ~GPUJpegBase() noexcept;
	virtual bool set_param(const core::Parameter& parameter) override;
protected:
	GPUJpegBase(log::Log &_log, core::pwThreadBase parent, const std::string& name);
	virtual bool init_device();
	uint16_t device;

};

} /* namespace io */
} /* namespace yuri */
#endif /* GPUJPEGBASE_H_ */
