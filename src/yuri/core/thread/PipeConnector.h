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


class EXPORT PipeConnector {

public:
								PipeConnector() = default;
								PipeConnector(pwPipeNotifiable thread, pwPipeNotifiable thread_src);
								PipeConnector(pPipe pipe, pwPipeNotifiable thread, pwPipeNotifiable thread_src);
								PipeConnector(PipeConnector&& orig) noexcept;
								PipeConnector(const PipeConnector&) = delete;
								~PipeConnector() noexcept;
	PipeConnector&				operator=(PipeConnector&&) noexcept;
	PipeConnector&				operator=(const PipeConnector&) = delete;
	pPipe			 			operator->();
	void 						reset();
	void 						reset(pPipe pipe);
								operator bool() const;
								operator pPipe();
	pPipe  						get() const;
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
