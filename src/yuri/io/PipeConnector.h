/*
 * PipeConnector.h
 *
 *  Created on: Aug 8, 2010
 *      Author: neneko
 */

#ifndef PIPECONNECTOR_H_
#define PIPECONNECTOR_H_
#include "BasicPipe.h"
#include "yuri/threads/ThreadBase.h"
#include <boost/smart_ptr.hpp>

namespace yuri {

namespace io {

using namespace yuri::threads;
using boost::shared_ptr;
using boost::weak_ptr;

class BasicIOThread;

class PipeConnector {
private:
	PipeConnector() {}
public:
	PipeConnector(weak_ptr<ThreadBase> thread);
	PipeConnector(shared_ptr<BasicPipe> pipe, weak_ptr<ThreadBase> thread);
	PipeConnector(const PipeConnector& orig);
	virtual ~PipeConnector();
	shared_ptr<BasicPipe> operator->();
	void reset();
	void reset(shared_ptr<BasicPipe> pipe);
	operator bool();
	operator shared_ptr<BasicPipe>();
	BasicPipe * get();
protected:
	shared_ptr<BasicPipe> pipe;
	weak_ptr<ThreadBase> thread;
	void set_pipe(shared_ptr<BasicPipe> pipe);

};

}

}

#endif /* PIPECONNECTOR_H_ */
