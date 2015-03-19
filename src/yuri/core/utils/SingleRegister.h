/*!
 * @file 		SingleRegister.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		01.03.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef SINGLEREGISTER_H_
#define SINGLEREGISTER_H_

#include <unordered_map>
#include <vector>
#include <stdexcept>
namespace yuri {
namespace core {
namespace generator {

template<class Key,
class Value>
struct SingleRegister {
	typedef Key					key_type;
	typedef Value 				value_type;

	typedef std::unordered_map<key_type, value_type>
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

	value_type		find_value (const key_type& key) const {
		auto it = map_.find(key);
		if (it == map_.end()) throw std::runtime_error("Key not found");
		return it->second;
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



#endif /* SINGLEREGISTER_H_ */
