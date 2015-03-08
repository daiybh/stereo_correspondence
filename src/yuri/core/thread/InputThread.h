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

void print_cfgs(log::Log& log, log::debug_flags flags, const core::InputDeviceInfo& info);



namespace detail {

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

bool is_last_key(const std::vector<core::InputDeviceConfig>& cfgs, const std::string& key, const std::vector<std::string>& used)
{
	for (const auto& c: cfgs) {
		for (const auto& p: c.params) {
			if (p.first != key && !contains(used, p.first)) {
				return false;
			}
		}
	}
	return true;
}

std::vector<std::string> unused_keys(const core::InputDeviceConfig& cfg, const std::vector<std::string>& used)
{
	std::vector<std::string> keys;
	for (const auto& p: cfg.params) {
		if (!contains(used, p.first)) {
			keys.push_back(p.first);
		}
	}
	return keys;
}

void print_cfgs_k(log::Log& log, log::debug_flags flags, const std::vector<core::InputDeviceConfig>& cfgs, std::vector<std::string> order, std::vector<std::string> used, std::string prefix)
{
//	log[log::info] << "pcfg_k";
	if (cfgs.empty()) return;
	if (order.empty()) {
		// We used up all keys and there's still some params, so lets print them here
		for (const auto& cfg: cfgs) {
			auto keys = unused_keys(cfg, used);
			auto l = log[flags] << prefix;
			for (const auto& key: keys) {
				l << "key: " << cfg.params[key].get<std::string>() << ", ";
			}
		}
		return;
	}
	const auto& key = order[0];
	auto used_next = used;
	used_next.push_back(key);
	if (is_last_key(cfgs, key, used)) {
		// We have last key, so let's use short output
		auto l = log[flags] << prefix << key << ": ";
		core::utils::print_list(l, find_map_key_values<std::string>(cfgs, key));
	} else {
		for (const auto& c: find_submap_by_key<std::string>(cfgs, key)) {
			log[flags] << prefix << key << ": " << c.first;
			print_cfgs_k(log, flags, c.second, std::vector<std::string>(order.begin()+1, order.end()), used_next, prefix+"\t");
		}
	}
}

}

void print_cfgs(log::Log& log, log::debug_flags flags, const core::InputDeviceInfo& info)
{
	detail::print_cfgs_k(log, flags, info.configurations, info.main_param_order, std::vector<std::string>{}, "\t");
}


}
}



#endif /* INPUTTHREAD_H_ */
