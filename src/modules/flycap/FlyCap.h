/*!
 * @file 		FlyCap.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		02.02.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef FLYCAP_H_
#define FLYCAP_H_

#include "yuri/core/thread/IOThread.h"
#ifdef YURI_CYGWIN
#define YURI_FLYCAP_C_ONLY
#endif
#ifndef YURI_FLYCAP_C_ONLY
#include "FlyCapture2.h"
#else
#include "C/FlyCapture2_C.h"
#endif
namespace yuri {
namespace flycap {

class FlyCap: public core::IOThread
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	FlyCap(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~FlyCap() noexcept;
private:
	
	virtual void run() override;
	virtual bool set_param(const core::Parameter& param) override;
private:
	resolution_t resolution_;
	format_t format_;
	size_t fps_;

	size_t index_;
	size_t serial_;
	bool keep_format_;
#ifndef	YURI_FLYCAP_C_ONLY
	FlyCapture2::Camera camera_;
#else
	fc2Context ctx_;
#endif
	duration_t shutdown_delay_;
};

} /* namespace flycap */
} /* namespace yuri */
#endif /* FLYCAP_H_ */
