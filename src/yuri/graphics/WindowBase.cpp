/*!
 * @file 		WindowBase.cpp
 * @author 		Zdenek Travnicek
 * @date 		31.5.2008
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2008 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "WindowBase.h"

namespace yuri
{
namespace graphics
{

WindowBase::WindowBase(log::Log &log_, core::pwThreadBase parent, core::Parameters &p):
	ThreadBase(log_,parent),x(0),y(0),width(0),height(0),name("noname")
{
	params.merge(p);
}

WindowBase::~WindowBase()
{
}













}
}

// End of File
