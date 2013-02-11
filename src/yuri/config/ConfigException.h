#ifndef CONFIGEXCEPTION_H_
#define CONFIGEXCEPTION_H_
#include "yuri/exception/Exception.h"

namespace yuri
{
namespace config
{
	
class ConfigException: yuri::exception::Exception
{
public:
	ConfigException();
	virtual ~ConfigException() throw();
};

}
}

#endif /*CONFIGEXCEPTION_H_*/
