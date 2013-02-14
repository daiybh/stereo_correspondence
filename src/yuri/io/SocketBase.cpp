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
		pause_timer();
		sleep(50000);
		start_timer();
	}	
}

}

}
