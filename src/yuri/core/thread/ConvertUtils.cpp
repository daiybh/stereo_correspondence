/*
 * ConvertUtils.cpp
 *
 *  Created on: 30.10.2013
 *      Author: neneko
 */

#include "ConvertUtils.h"
#include <unordered_map>
#include <queue>
//#include <iostream>
namespace yuri {
namespace core {


namespace {

mutex	path_cache_mutex;
std::unordered_map<converter_key, convert::path_list> path_cache;


}

// Searches all convertors using Dijkstra algorithm
convert::path_list find_conversion(format_t format_in, format_t format_out)
{
	converter_key search_key{format_in, format_out};
	if (format_in == format_out) return {};
	{
		lock_t _(path_cache_mutex);
		auto pit = path_cache.find(search_key);
		if (pit != path_cache.end()) {
			return pit->second;
		}
	}
	std::unordered_map<format_t, bool> visited;
	std::unordered_map<converter_key, value_type > best_convertor;
	std::unordered_multimap<format_t, converter_key> starts;
	std::unordered_map<format_t, size_t> costs;
	std::unordered_map<format_t, convert::path_list> paths;
	auto cmp = [&best_convertor](const converter_key& a, const converter_key& b)
					{return best_convertor[a].second < best_convertor[b].second;};
	std::priority_queue<converter_key, std::vector<converter_key>, decltype(cmp)	>
							stack(cmp);
	const auto& conv = core::ConverterRegister::get_instance();
	const auto& keys = conv.list_keys();

	costs[format_in] = 1; // Default cost
	paths[format_in] = {}; // Empty path by default

	// Prepare the graph

	for (const auto& k: keys) {
		visited[k.second]=false; // By default no convertor is visited
		starts.emplace(k.first, k); // Prepare all convertors
		auto vals = conv.find_value(k); // Select the best convertor for each format pair
		for (const auto& v: vals) {
			if (best_convertor.find(k) == best_convertor.end()) {
				best_convertor[k]=v;
			} else {
				if (v.second < best_convertor[k].second) {
					best_convertor[k]=v;
				}
			}
		}
	}

	// Populate stack with initial edges
	visited[format_in] = true;
	auto er = starts.equal_range(format_in);
	auto it = er.first;
	while(it!=er.second) {
		stack.emplace(it++->second);
	}


//	std::cout << "Looking up " << format_in << " -> " << format_out << "\n";
//	std::cout << "Prepared " << best_convertor.size() << " convertors\n";
	while (!stack.empty()) {
//		std::cout << "Stack size "<<stack.size() <<"\n";
		auto head = stack.top();
		stack.pop();

		visited[head.second]=true;
		size_t cost_start = costs[head.first];
		size_t cost_end = costs[head.second];
		const auto& bc = best_convertor[head];
		size_t new_cost = cost_start + bc.second;
//		std::cout << "Testing " << head.first << " -> " << head.second << ", old cost " << cost_end << ", new: " << new_cost << "\n";
		if (cost_end > 0 && cost_end < new_cost) continue; // Ok, this way is not interesting

		costs[head.second] = new_cost;

		auto v = paths[head.first];
		v.push_back({bc.first, head.first, head.second});
		paths[head.second]=std::move(v);

		auto er = starts.equal_range(head.second);
		auto it = er.first;
		while(it!=er.second) {
			if (!visited[it->second.second]) stack.emplace(it->second);
			++it;
		}
	}

	{
		lock_t _(path_cache_mutex);
		auto pit = path_cache.find(search_key);
		if (pit != path_cache.end()) {
			path_cache[search_key]=paths[format_out];
		}
	}
//	std::cout << "Final cost " << costs[format_out] << "\n";
	return paths[format_out];
}


}
}

