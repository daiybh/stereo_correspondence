/*!
 * @file 		DeltaInput.h
 * @author 		Zdenek Travnicek
 * @date 		11.2.2013
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef DeltaInput_H_
#define DeltaInput_H_

#include "yuri/core/IOThread.h"
#include "VideoMasterHD_Core.h"
#include "VideoMasterHD_Sdi.h"


namespace yuri {
namespace delta {

class DeltaInput: public yuri::core::IOThread
{
public:
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();
	virtual ~DeltaInput();
private:
	DeltaInput(log::Log &log_, core::pwThreadBase parent,core::Parameters &parameters);
	virtual void run();
	virtual bool set_param(const core::Parameter& param);
	void throw_call(ULONG res, std::string msg);
	format_t format;

	HANDLE delta_handle_;
};

} /* namespace dummy_module */
} /* namespace yuri */
#endif /* DeltaInput_H_ */
