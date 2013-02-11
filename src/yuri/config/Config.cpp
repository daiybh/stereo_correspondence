#include "Config.h"

namespace yuri
{
namespace config
{

using namespace yuri::log;


Config* Config::default_config=0;
boost::mutex Config::default_config_mutex;
std::string Config::cf("yuri.config_file"), Config::cfs("yuri.config_files");

Config* Config::get_config(int /*index*/)
{
	boost::mutex::scoped_lock l(default_config_mutex);
	if (!default_config) {
		l.unlock();
		Log _log(std::cerr);
		default_config = new Config(_log);
	}
	return default_config;
}


Config::Config(Log &log_):log(log_)
{
	boost::mutex::scoped_lock l(default_config_mutex);
	boost::mutex::scoped_lock ll(config_mutex);
	if (!default_config) default_config=this;
	log.setLabel("[Config] ");
	if (get_env(cf)) read_config_file(get_env(cf));
}

Config::Config(Log &log_, const char *confname):log(log_)
{
	boost::mutex::scoped_lock l(default_config_mutex);
	boost::mutex::scoped_lock ll(config_mutex);
	if (!default_config) default_config=this;
	log.setLabel("[Config] ");
	if (get_env(cf)) read_config_file(get_env(cf));
	ll.unlock();
	init_config(1,(char**)&confname);
}


Config::~Config()
{
	boost::mutex::scoped_lock l(config_mutex);
	/*while (!configs.empty()) {
		//delete *(configs.begin());
		configs.pop_front();
	}

	for (std::map<std::string,Callback*>::iterator i=callbacks.begin();i!=callbacks.end();++i) {
		log[verbose_debug] << "Deleting callback " << (*i).first << std::endl;
		if ((*i).second) delete (*i).second;
	}*/
}
/// Function to read configuration files
///
/// It should look for configuration files in this order:
/// 0) configuration file provided by cmd line option (not supported yet)
/// 1) ./<appname>.conf
/// 2) ./.<appname>.conf
/// 3) ~/.<appname>.conf
/// 4) ./yuri.conf
/// 5) ./.yuri.conf
/// 6) ~/.yuri.conf

bool Config::init_config(int /*argc*/, char **/*argv*/, bool /*use_file*/)
{
	boost::mutex::scoped_lock l(config_mutex);
	bool ret=true;

	// Skipping command line argument parsing

	if (!ret) return ret;  // If an error occures during command line expansion, we're gonna return immediately
	ret = false;
#ifdef YURI_HAVE_LIBCONFIG
	std::string home_dir(getenv("HOME"));
	boost::filesystem::path lf(argv[0]);
	std::string appconf=lf.leaf().native()+".conf",def_conf_name="yuri.conf";

	if (read_config_file("./"+appconf)) ret = true;
	if (read_config_file("./."+appconf)) ret = true;
	if (read_config_file(home_dir+"/."+appconf)) ret = true;
	if (read_config_file("./"+def_conf_name)) ret = true;
	if (read_config_file("./."+def_conf_name)) ret = true;
	if (read_config_file(home_dir+"/."+def_conf_name)) ret = true;
#endif
	return ret;

}

bool Config::read_config_file(std::string filename)
{
#ifdef YURI_HAVE_LIBCONFIG
	log[debug] << "Reading config file " << filename << std::endl;
	shared_ptr<libconfig::Config> cfg (new libconfig::Config());
	try {
		cfg->readFile(filename.c_str());
	}
	catch (libconfig::FileIOException) {
		log[warning] << "File exception occured, configuration file " << filename << " is probably inaccessible" << std::endl;
		//delete cfg;
		return false;
	}
	catch (libconfig::ParseException &e) {
		log[error] << "Parse exception occured!" << std::endl << "Parser returned: " << e.getError() << " on line " << e.getLine() << std::endl;
		//delete cfg;
		return false;
	}
	log[info] << "Successfully parsed file " << filename << std::endl;
	configs.push_back(cfg);
	std::string fn;
	if (cfg->exists(cf)) {
		try {
			cfg->lookupValue(cf,fn);
			read_config_file(fn);
		}
		catch (libconfig::SettingNotFoundException) {
		}
	}
	if (cfg->exists(cfs)) {
		int fn_len=cfg->lookup(cfs).getLength();
		for (int i = 0; i < fn_len; ++i) {
			try {
				fn = (const char *) cfg->lookup(cfs)[i];
				read_config_file(fn);
			}
			catch (libconfig::SettingNotFoundException) {
			}
		}
	}
#else
	log[warning] << "NOT reading config file " << filename << ", missing libconfig support\n";
	return false;
#endif
	return true;
}

bool Config::exists(std::string /*path*/)
{
	boost::mutex::scoped_lock l(config_mutex);
#ifdef YURI_HAVE_LIBCONFIG
	if (configs.empty()) throw (ConfigException());
	for (std::deque<shared_ptr<libconfig::Config> >::iterator i=configs.begin();
			i!=configs.end();++i) {
		if ((*i)->exists(path)) return true;
	}
#else
	throw (ConfigException());
#endif
	return false;
}

int Config::get_array_size(std::string &/*path*/)
{
	boost::mutex::scoped_lock l(config_mutex);
#ifdef YURI_HAVE_LIBCONFIG
	if (configs.empty()) throw (ConfigException());
	try {
		for (std::deque<shared_ptr<libconfig::Config> >::iterator i=configs.begin();
					i!=configs.end();++i) {
			if ((*i)->exists(path)) return (*i)->lookup(path).getLength();
		}
	}
	catch (libconfig::SettingNotFoundException) {
		log[warning] << "Configuration option " << path << " not found" << std::endl;
	}
	catch (libconfig::SettingTypeException) {
		log[warning] << "Configuration option " << path << " is not array" << std::endl;
	}
#else
	throw (ConfigException());
#endif
	return -1;
}


void Config::set_callback(std::string name, pCallback func, pThreadBase data)
{
	boost::mutex::scoped_lock l(config_mutex);
	shared_ptr<Callback> c (new Callback(func,data));
	callbacks[name] = c;
}

shared_ptr<Callback> Config::get_callback(std::string name)
{
	boost::mutex::scoped_lock l(config_mutex);
	return callbacks[name];
}

const char *Config::get_env(std::string path)
{
	const char *out;
	std::string tmp;
	out = getenv(path.c_str());
	if (out) return out;

	tmp="";
	std::transform(path.begin(),path.end(),tmp.begin(),(int(*)(int)) std::toupper);
	out = getenv(tmp.c_str());
	if (out) return out;

	tmp = path;
	std::replace(tmp.begin(),tmp.end(),'.','_');
	out = getenv(tmp.c_str());
	if (out) return out;

	std::transform(path.begin(),path.end(),tmp.begin(),(int(*)(int)) std::toupper);
	std::replace(tmp.begin(),tmp.end(),'.','_');
	out = getenv(tmp.c_str());
	if (out) return out;

	return 0;
}

}
}

// End of File
