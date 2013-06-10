/*!
 * @file 		Config.h
 * @author 		Zdenek Travnicek
 * @date 		31.5.2008
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2008 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef CONFIG_H_
#define CONFIG_H_
#include "yuri/log/Log.h"
#include <cstdlib>
#include <string>
#include <map>
#include <deque>
#include <algorithm>
#include <cctype>
#include "ConfigException.h"
#include "yuri/exception/InitializationFailed.h"
#include "yuri/exception/OutOfRange.h"
#include "Callback.h"


namespace yuri
{
namespace core
{

class EXPORT Config
{
public:
								Config(Log &log);
								Config(Log &log, const char *confname);
	virtual 					~Config();
	bool 						init_config(int argc, char **argv, bool use_file=true);
	int 						get_array_size(std::string &path);
	bool 						exists(std::string path);
	void 						set_callback(std::string name, pCallback func, pThreadBase data);
	shared_ptr<Callback> 		get_callback(std::string name);
	template<typename T> bool 	get_value(std::string path, T &out, T def);
	template<typename T> bool 	get_value(std::string path, T &out);
	template<typename T> bool 	get_value_from_array(std::string path, int index, T &out);
protected:
	bool 						read_config_file(std::string filename);
	const char* 				get_env(std::string path);
	log::Log					log;
	boost::mutex 				config_mutex;
	std::map<std::string,shared_ptr<Callback> >
								callbacks;

public: static Config* 			get_config(int index=0);
protected: static Config* 		default_config;
protected: static boost::mutex 	default_config_mutex;
protected: static std::string 	cf;
		   static std::string 	cfs;
};

template<typename T>  bool
	Config::get_value(std::string path, T &out, T def)
{
	if (get_value(path, out)) return true;
	out = def;
	return false;
}
template<typename T>  bool
	Config::get_value(std::string /*path*/, T &/*out*/)
{
	boost::mutex::scoped_lock l(config_mutex);
	return false;
}

template<>  inline bool
	Config::get_value<std::string> (std::string path, std::string &out)

{
	boost::mutex::scoped_lock l(config_mutex);
	const char * env = get_env(path);
	if (env) {
		out = env;
		return true;
	}
	return false;
}

template<typename T>  bool
	Config::get_value_from_array(std::string /*path*/, int /*index*/, T &/*out*/)
{
	boost::mutex::scoped_lock l(config_mutex);
	return false;
}


template<>  inline bool
	Config::get_value_from_array<std::string> (std::string /*path*/, int /*index*/, std::string &/*out*/)
{
	boost::mutex::scoped_lock l(config_mutex);

	return false;
}
}
}
#endif /*CONFIG_H_*/
