/*!
 * @file 		SocketBase.h
 * @author 		Zdenek Travnicek
 * @date 		27.10.2008
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2008 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef SOCKETBASE_H_
#define SOCKETBASE_H_
#include "yuri/core/ThreadBase.h"
namespace yuri
{

namespace core
{
class SocketBase: public ThreadBase
{
public:
								SocketBase(log::Log &_log, pwThreadBase parent);
	virtual 					~SocketBase();
	virtual yuri::size_t 		read(yuri::ubyte_t * data, yuri::size_t size)=0;
	virtual yuri::size_t 		write(yuri::ubyte_t * data,yuri::size_t size)=0;
	virtual bool 				data_available() {return false;}
	virtual int 				get_fd() = 0;
protected:
	void 						run();
};

}

}

#endif /*SOCKETBASE_H_*/
