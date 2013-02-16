/*!
 * @file 		ConfigException.h
 * @author 		Zdenek Travnicek
 * @date 		31.5.2008
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2008 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef CONFIGEXCEPTION_H_
#define CONFIGEXCEPTION_H_
#include "yuri/exception/Exception.h"

namespace yuri
{
namespace config
{
	
class EXPORT ConfigException: yuri::exception::Exception
{
public:
	ConfigException();
	virtual ~ConfigException() throw();
};

}
}

#endif /*CONFIGEXCEPTION_H_*/
