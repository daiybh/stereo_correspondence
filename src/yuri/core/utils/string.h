/*!
 * @file 		string.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		07.06.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef STRING_H_
#define STRING_H_

#include <string>
#include <vector>
#include <cstring>
#include <algorithm>

namespace yuri {
namespace core {
namespace utils {

template<class It, class CharT>
std::vector<std::basic_string<CharT>> split_string(It first, It last, const CharT& delimiter)
{
	using str_type = std::basic_string<CharT>;
	std::vector<str_type> parts;
	if (first == last) return parts;
	while (first != last) {
		auto pos = std::find(first, last, delimiter);
		parts.push_back(str_type(first, pos));
		first = pos;
		if (first != last) ++first;
	}
	return parts;
}

template<class CharT>
std::vector<std::basic_string<CharT>> split_string(const CharT* str, const CharT& delimiter)
{
	return split_string(str, str+std::strlen(str), delimiter);
}

template<class CharT>
std::vector<std::basic_string<CharT>> split_string(const std::basic_string<CharT>& str, const CharT& delimiter)
{
	return split_string(str.begin(), str.end(), delimiter);
}


}
}
}


#endif /* STRING_H_ */
