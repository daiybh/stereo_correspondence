#ifndef NULL_H_
#define NULL_H_
#include <yuri/config/RegisteredClass.h>

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
