/*!
 * @file 		MultiRegister.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		29.10.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef BASICMULTIREGISTER_H_
#define BASICMULTIREGISTER_H_

#include <unordered_map>
#include <vector>
namespace yuri {
namespace core {
namespace generator {

template<class Key,
class Value>
struct MultiRegister {
	typedef Key					key_type;
	typedef Value 				value_type;

	typedef std::unordered_multimap<key_type, value_type>
								map_type;

	typedef typename map_type::iterator
								iterator;

	typedef typename map_type::const_iterator
								const_iterator;

	void 						add_value(const key_type& key, const value_type& value) {
		map_.emplace(key, value);
	}
	void 						remove_value(const key_type& /*key*/, const value_type& /*value*/) {
		// TODO Implement this
	}

	std::vector<value_type>		find_value (const key_type& key) const {
		std::vector<value_type> values;
		auto range = map_.equal_range(key);
		auto it = range.first;
		if (range.first == map_.end()) return values;
		while (it != range.second) {
			values.push_back(it++->second);
		}
		return values;
	}

	std::vector<key_type>		list_keys() const {
		std::vector<key_type>	keys;
		typename std::vector<key_type>::const_iterator last;
		for (auto it: map_) {
			if (keys.empty() || it.first != *last) {
				last = keys.emplace(keys.end(), it.first);
			}
		}
		return keys;
	}

	iterator					begin() { return map_.begin(); }
	const_iterator				begin() const { return map_.begin(); }
	const_iterator				cbegin() const { return map_.cbegin(); }

	iterator					end() { return map_.end(); }
	const_iterator				end() const { return map_.end(); }
	const_iterator				cend() const { return map_.cend(); }
private:
	map_type 					map_;

};


}
}

}



#endif /* BASICMULTIREGISTER_H_ */
