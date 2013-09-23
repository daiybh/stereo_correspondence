/*
 * utils.h
 *
 *  Created on: 9.6.2013
 *      Author: neneko
 */

#ifndef UTILS_H_
#define UTILS_H_
#include <map>
#include <vector>
#include <sstream>
#include <cctype>
#include <mutex>
namespace yuri {

/*!
 * @brief Ancillary class for initializing std::map
 *
 * Usage std::map<A, B> name_of_map = map_list_of<A, B>(a0, b0)(a1, b1)(a2, b2);
 */
template<class Key, class Value> class map_list_of {
public:
	operator std::map<Key, Value>() { return tmp_map; }
	map_list_of() {}
	map_list_of(const Key& key, const Value& value) {tmp_map[key]=value;}
	map_list_of & operator() (const Key& key, const Value& value) {tmp_map[key]=value;return *this;}
protected:
	std::map<Key, Value> tmp_map;
};

template<class Value> class list_of {
public:
	operator std::vector<Value>() { return tmp_vec; }
	list_of() {}
	list_of(const Value& val) { tmp_vec.push_back(val); }
	list_of & operator() (const Value& val) {tmp_vec.push_back(val); return *this;}
protected:
	std::vector<Value> tmp_vec;
};

struct bad_lexical_cast: public std::runtime_error {
	bad_lexical_cast(const std::string& str):runtime_error(str) {}
};


template<class T, class U>
typename std::enable_if<!std::is_convertible<U, T>::value, T>::type
lexical_cast(const U& val)
{
	T outval;
	std::stringstream str;
	str << val;
	str >> outval;
	if (str.fail()) throw bad_lexical_cast("Bad lexical cast");
	return outval;
}

template<class T, class U>
typename std::enable_if<std::is_convertible<U, T>::value, T>::type
lexical_cast(const U& val)
{
	return val;
}

template<class Char, class traits>
bool iequals(const std::basic_string<Char, traits>& a, const std::basic_string<Char, traits>& b)
{
	if (a.size() != b.size()) return false;
	auto ai = a.cbegin();
	auto bi = b.cbegin();
	while (ai!=a.cend()) {
		if (std::toupper(*ai++)!=std::toupper(*bi++)) return false;
	}
	return true;
}
template<class Char, class traits>
bool iequals(const std::basic_string<Char, traits>& a, const Char *b)
{
	return iequals(a, std::basic_string<Char, traits>(b));
}


template<class T>
class SingletonBase: public T {
public:
	static T& get_instance() {
		static T instance;
		return instance;
	}

private:
	SingletonBase() {};
	virtual ~SingletonBase() {}
};
}


#endif /* UTILS_H_ */
