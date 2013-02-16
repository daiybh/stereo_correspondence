/*!
 * @file 		Null.h
 * @author 		Zdenek Travnicek
 * @date 		24.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef NULL_H_
#define NULL_H_
#include "yuri/config/RegisteredClass.h"

namespace yuri
{

namespace io
{
using namespace yuri::log;
class Null: public BasicIOThread
{
public:
	IO_THREAD_GENERATOR_DECLARATION
	static shared_ptr<Parameters> configure();
	virtual ~Null();
protected:
	Null(Log &_log,pThreadBase parent, Parameters& parameters) IO_THREAD_CONSTRUCTOR;
	virtual bool step();
};

}

}

#endif /*NULL_H_*/
