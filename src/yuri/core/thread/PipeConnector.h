/*!
 * @file 		PipeConnector.cpp
 * @author 		Zdenek Travnicek
 * @date 		8.8.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
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
								PipeConnector(pwPipeNotifiable thread);
								PipeConnector(pPipe pipe, pwPipeNotifiable thread);
								PipeConnector(const PipeConnector& orig);
								~PipeConnector() noexcept;
	pPipe			 			operator->();
	void 						reset();
	void 						reset(pPipe pipe);
								operator bool() const;
								operator pPipe();
	pPipe  						get() const;
private:
	void 						set_pipe(pPipe pipe);
	void						set_notifications(pwPipeNotifiable) noexcept;
	pwPipeNotifiable			notifiable_;
	pPipe				 		pipe_;



};

}

}

#endif /* PIPECONNECTOR_H_ */
