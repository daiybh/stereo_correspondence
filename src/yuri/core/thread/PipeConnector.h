/*!
 * @file 		PipeConnector.cpp
 * @author 		Zdenek Travnicek
 * @date 		8.8.2010
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2010 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef PIPECONNECTOR_H_
#define PIPECONNECTOR_H_
#include "yuri/core/forward.h"
#include "yuri/core/pipe/PipeNotification.h"
namespace yuri {

namespace core {


class PipeConnector {
public:
	EXPORT 						PipeConnector() = default;
	EXPORT 						PipeConnector(pwPipeNotifiable thread, pwPipeNotifiable thread_src);
	EXPORT 						PipeConnector(pPipe pipe, pwPipeNotifiable thread, pwPipeNotifiable thread_src);
	EXPORT 						PipeConnector(PipeConnector&& orig) noexcept;
	EXPORT 						PipeConnector(const PipeConnector&) = delete;
	EXPORT 						~PipeConnector() noexcept;
	EXPORT PipeConnector&		operator=(PipeConnector&&) noexcept;
	EXPORT PipeConnector&		operator=(const PipeConnector&) = delete;
	EXPORT pPipe				operator->();
	EXPORT void 				reset();
	EXPORT void 				reset(pPipe pipe);
	EXPORT 						operator bool() const;
	EXPORT 						operator pPipe();
	EXPORT pPipe  				get() const;
private:
	void 						set_pipe(pPipe pipe);
	void						set_notifications(pwPipeNotifiable, pwPipeNotifiable) noexcept;
	pwPipeNotifiable			notifiable_;
	pwPipeNotifiable			notifiable_src_;
	pPipe				 		pipe_;



};

}

}

#endif /* PIPECONNECTOR_H_ */
