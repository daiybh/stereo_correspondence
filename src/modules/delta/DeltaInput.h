/*!
 * @file 		DeltaInput.h
 * @author 		Zdenek Travnicek
 * @date 		11.2.2013
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef DeltaInput_H_
#define DeltaInput_H_

#include "yuri/core/thread/IOThread.h"
#include "VideoMasterHD_Core.h"
#include "VideoMasterHD_Sdi.h"


namespace yuri {
namespace delta {

class DeltaInput: public yuri::core::IOThread
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	DeltaInput(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~DeltaInput() noexcept;
private:
	virtual void run() override;
	virtual bool set_param(const core::Parameter& param) override;
	void throw_call(ULONG res, std::string msg);
// Currently unused
	//	format_t format;

	HANDLE delta_handle_;
};

} /* namespace dummy_module */
} /* namespace yuri */
#endif /* DeltaInput_H_ */
