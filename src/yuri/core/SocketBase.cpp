/*!
 * @file 		SocketBase.cpp
 * @author 		Zdenek Travnicek
 * @date 		27.10.2008
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2008 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "SocketBase.h"

namespace yuri
{

namespace core
{

SocketBase::SocketBase(log::Log &_log, pwThreadBase parent):ThreadBase(_log,parent)
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
