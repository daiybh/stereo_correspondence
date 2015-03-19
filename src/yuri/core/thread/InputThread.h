/*!
 * @file 		InputThread.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		01.03.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef INPUTTHREAD_H_
#define INPUTTHREAD_H_

#include "yuri/core/utils/make_list.h"
#include "yuri/core/parameter/Parameters.h"
#include "yuri/log/Log.h"
namespace yuri {
namespace core{


struct InputDeviceConfig {
	// Resolution and format are here just for info...
	resolution_t resolution;
	format_t color_format;

	Parameters params;
};
struct InputDeviceInfo {
	std::string class_name;
	std::string device_name;
	std::vector<InputDeviceConfig> configurations;
	std::vector<std::string> main_param_order;
};

EXPORT void print_cfgs(log::Log& log, log::debug_flags flags, const core::InputDeviceInfo& info);



namespace detail {

EXPORT bool is_last_key(const std::vector<core::InputDeviceConfig>& cfgs, const std::string& key, const std::vector<std::string>& used);
EXPORT std::vector<std::string> unused_keys(const core::InputDeviceConfig& cfg, const std::vector<std::string>& used);
EXPORT void print_cfgs_k(log::Log& log, log::debug_flags flags, const std::vector<core::InputDeviceConfig>& cfgs, std::vector<std::string> order, std::vector<std::string> used, std::string prefix);



template<class Value>
std::vector<Value> find_map_key_values(const std::vector<core::InputDeviceConfig>& cfgs, const std::string& key)
{
	std::vector<Value> values;
	for (const auto& cfg: cfgs) {
		try {
			auto p = cfg.params[key];
			auto val = p.get<Value>();
			if (!contains(values, val)) {
				values.push_back(val);
			}
		}
		catch(std::exception&){}
	}
	return values;
}

template<class Value>
std::vector<core::InputDeviceConfig> find_submap_by_key_value(const std::vector<core::InputDeviceConfig>& cfgs, const std::string& key, const Value& value)
{
	std::vector<core::InputDeviceConfig> outcfg;
	for (const auto& cfg: cfgs) {
		try {
			auto p = cfg.params[key];
			auto val = p.get<Value>();
			if (val == value) {
				outcfg.push_back(cfg);
			}
		}
		catch(std::exception&){}
	}
	return outcfg;
}

template<class Value>
std::map<Value, std::vector<core::InputDeviceConfig>> find_submap_by_key(const std::vector<core::InputDeviceConfig>& cfgs, const std::string& key)
{
	auto vals = find_map_key_values<Value>(cfgs, key);
	std::map<Value, std::vector<core::InputDeviceConfig>> outmap;
	for (const auto& val: vals) {
		outmap[val] = find_submap_by_key_value(cfgs, key, val);
	}
	return outmap;
}
}
}
}



#endif /* INPUTTHREAD_H_ */
