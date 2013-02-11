/*
 * Split.h
 *
 *  Created on: Mar 30, 2011
 *      Author: worker
 */

#ifndef SPLIT_H_
#define SPLIT_H_

#include "yuri/io/BasicIOThread.h"
#include "yuri/config/RegisteredClass.h"


namespace yuri {

namespace io {

using namespace yuri::config;
using boost::shared_array;
class Split: public BasicIOThread {
public:
	Split(Log &_log, pThreadBase parent, Parameters &params);
	virtual ~Split();
	static shared_ptr<BasicIOThread> generate(Log &_log,pThreadBase parent,
			Parameters& parameters) throw (Exception);
	static shared_ptr<Parameters> configure();
protected:
	virtual bool step();
};

}

}

#endif /* SPLIT_H_ */
