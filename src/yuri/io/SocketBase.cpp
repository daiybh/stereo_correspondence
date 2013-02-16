#include "SocketBase.h"

namespace yuri
{

namespace io
{

SocketBase::SocketBase(Log &_log, pThreadBase parent):ThreadBase(_log,parent)
{
}

SocketBase::~SocketBase()
{
}

void SocketBase::run()
{
	while(still_running()) {
		sleep(50000);
	}	
}

}

}
