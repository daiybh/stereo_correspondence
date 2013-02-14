#ifndef SOCKETBASE_H_
#define SOCKETBASE_H_
#include "yuri/threads/ThreadBase.h"
namespace yuri
{

namespace io
{
using namespace yuri::log;
using namespace yuri::threads;
class SocketBase: public ThreadBase
{
public:
	SocketBase(Log &_log, pThreadBase parent);
	virtual ~SocketBase();
	virtual yuri::size_t read(yuri::ubyte_t * data, yuri::size_t size)=0;
	virtual yuri::size_t write(yuri::ubyte_t * data,yuri::size_t size)=0;
	virtual bool data_available() {return false;}
	virtual int get_fd() = 0;
protected:
	void run();
};

}

}

#endif /*SOCKETBASE_H_*/
