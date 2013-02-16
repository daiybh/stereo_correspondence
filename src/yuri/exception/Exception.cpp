/*!
 * @file 		Exception.cpp
 * @author 		Zdenek Travnicek
 * @date 		31.5.2008
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2008 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "Exception.h"

namespace yuri
{
namespace exception
{

Exception::Exception():reason("Generic exception")
{
}

Exception::Exception(std::string reason):reason(reason)
{

}
Exception::~Exception() throw()
{
}

const char *Exception::what() const throw()
{
	return (const char*) reason.c_str();
}

}
}

// End of File
