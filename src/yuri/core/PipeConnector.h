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
//#include "BasicPipe.h"
//#include "yuri/core/ThreadBase.h"
#include "yuri/core/forward.h"

namespace yuri {

namespace core {


class EXPORT PipeConnector {
private:
								PipeConnector() {}
public:
								PipeConnector(pwThreadBase thread);
								PipeConnector(pBasicPipe pipe, pwThreadBase thread);
								PipeConnector(const PipeConnector& orig);
	virtual 					~PipeConnector();
	pBasicPipe			 		operator->();
	void 						reset();
	void 						reset(pBasicPipe pipe);
								operator bool();
								operator shared_ptr<BasicPipe>();
	BasicPipe * 				get();
protected:
	pBasicPipe			 		pipe;
	pwThreadBase		 		thread;
	void 						set_pipe(pBasicPipe pipe);

};

}

}

#endif /* PIPECONNECTOR_H_ */
