/*!
 * @file 		Split.h
 * @author 		Zdenek Travnicek
 * @date 		30.3.2011
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2011 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef SPLIT_H_
#define SPLIT_H_

#include "yuri/io/BasicIOThread.h"
#include "yuri/config/RegisteredClass.h"


namespace yuri {

namespace io {

using namespace yuri::config;

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
